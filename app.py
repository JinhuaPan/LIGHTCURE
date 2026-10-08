# -*- coding: utf-8 -*-
"""
LIGHTCURE - 术后动态监测决策支持系统（v4）
========================================
后端：术后动态模型（T0 / T1 / T3）
- T0: 术前 11 + IHC
- T1: 术前 11 + 术后 3 月变量 + IHC
- T3: 术前 11 + 术后 3 月 + 6 月变量 + IHC

IHC 和术后变量都允许缺失（走缺失嵌入）。
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import io
from datetime import datetime
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="LIGHTCURE - Postoperative Dynamic Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700; color: #1a5276;
        text-align: center; padding: 1rem 0;
        border-bottom: 3px solid #2e86c1; margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem; color: #2c3e50;
        text-align: center; margin-bottom: 2rem;
    }
    .risk-high { background-color: #e74c3c; color: white; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-weight: 700; }
    .risk-intermediate { background-color: #f39c12; color: white; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-weight: 700; }
    .risk-low { background-color: #27ae60; color: white; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-weight: 700; }
    .recommend-ire { background-color: #2e86c1; color: white; padding: 0.5rem 1.5rem;
        border-radius: 10px; font-weight: 700; font-size: 1.2rem; text-align: center; }
    .recommend-rfa { background-color: #27ae60; color: white; padding: 0.5rem 1.5rem;
        border-radius: 10px; font-weight: 700; font-size: 1.2rem; text-align: center; }
    .recommend-either { background-color: #f39c12; color: white; padding: 0.5rem 1.5rem;
        border-radius: 10px; font-weight: 700; font-size: 1.2rem; text-align: center; }
    .heat-phenotype { background-color: #8e44ad; color: white; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-weight: 700; }
    .footer { text-align: center; color: #7f8c8d; font-size: 0.8rem;
        padding: 1rem 0; border-top: 1px solid #ecf0f1; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 模型定义（与术后动态模型一致，支持 postop + IHC 双缺失编码）
# ============================================================================

class MissingAdaptiveEncoder(nn.Module):
    def __init__(self, input_dim, preop_dim, postop_dim, ihc_dim=0,
                 hidden_dims=[192, 96, 48], dropout=0.45):
        super(MissingAdaptiveEncoder, self).__init__()

        self.input_dim = input_dim
        self.preop_dim = preop_dim
        self.postop_dim = postop_dim
        self.ihc_dim = ihc_dim

        if self.postop_dim > 0:
            self.missing_embeddings = nn.Parameter(
                torch.randn(self.postop_dim, 1) * 0.01
            )
        else:
            self.register_parameter('missing_embeddings', None)

        if self.ihc_dim > 0:
            self.ihc_missing_embeddings = nn.Parameter(
                torch.randn(self.ihc_dim, 1) * 0.01
            )
        else:
            self.register_parameter('ihc_missing_embeddings', None)

        self.bn_input = nn.BatchNorm1d(input_dim)
        self.dropout_input = nn.Dropout(0.25)

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            d_rate = min(0.55, 0.25 + i * 0.1)
            layers.append(nn.Dropout(d_rate))
            prev_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, postop_mask, ihc_mask=None):
        batch_size = x.size(0)
        x_preop = x[:, :self.preop_dim]

        if self.postop_dim > 0:
            x_postop = x[:, self.preop_dim:self.preop_dim + self.postop_dim]
            missing_emb = self.missing_embeddings.squeeze().unsqueeze(0).expand(batch_size, -1)
            x_postop = x_postop * postop_mask + (1 - postop_mask) * missing_emb
        else:
            x_postop = x[:, self.preop_dim:self.preop_dim]

        if self.ihc_dim > 0:
            x_ihc = x[:, self.preop_dim + self.postop_dim:]
            if ihc_mask is None:
                ihc_mask = torch.ones_like(x_ihc)
            ihc_emb = self.ihc_missing_embeddings.squeeze().unsqueeze(0).expand(batch_size, -1)
            x_ihc = x_ihc * ihc_mask + (1 - ihc_mask) * ihc_emb
        else:
            x_ihc = x[:, self.preop_dim + self.postop_dim:]

        parts = [x_preop]
        if self.postop_dim > 0:
            parts.append(x_postop)
        if self.ihc_dim > 0:
            parts.append(x_ihc)
        x_combined = torch.cat(parts, dim=1)

        x_combined = self.bn_input(x_combined)
        x_combined = self.dropout_input(x_combined)
        x_combined = self.hidden(x_combined)
        x_combined = self.output(x_combined)

        return self.sigmoid(x_combined).squeeze(-1)


class LIGHTCURE_Dynamic_Model(nn.Module):
    def __init__(self, input_dim, preop_dim, postop_dim, ihc_dim=0,
                 hidden_dims=[192, 96, 48], dropout=0.45):
        super(LIGHTCURE_Dynamic_Model, self).__init__()
        self.encoder = MissingAdaptiveEncoder(
            input_dim, preop_dim, postop_dim, ihc_dim, hidden_dims, dropout
        )

    def forward(self, x, postop_mask, ihc_mask=None):
        return self.encoder(x, postop_mask, ihc_mask)


# ============================================================================
# 变量定义（与术后动态模型完全一致）
# ============================================================================

PREOP_MANDATORY_VARS = [
    "us_well_defined_margin_preop",
    "Etiology",
    "us_enlarged_lymph_nodes_preop",
    "DWI_High_preop",
    "Restricted_Diffusion_preop",
    "Arterial_Enhancement_preop",
    "location",
    "us_gallbladder_invasion_preop",
    "Margin_Ill_Defined_preop",
    "Number_of_lesions",
    "Maximum_diameter",
]

POSTOP_VARS_T1 = [
    "New_Nodule_post3m",
    "New_Nodule_Size_mm_post3m",
    "New_Nodule_Enhancement_post3m",
    "New_Nodule_Count_post3m",
    "Margil_Enhancement_post3m",
    "Arterial_Enhancement_post3m",
]

POSTOP_VARS_T3 = [
    "Complete_Ablation_post6m",
    "Margil_Enhancement_post6m",
    "New_Nodule_post6m",
    "New_Nodule_Count_post6m",
    "New_Nodule_Size_mm_post6m",
    "New_Nodule_Enhancement_post6m",
    "Arterial_Enhancement_post6m",
]

IHC_ENHANCEMENT_VARS = [
    "HSP70", "HIF_1α", "BCL_2", "MVI", "E_cadherin", "CK19", "VEGF", "MMP_9"
]

# T3 是 T1 + T3 的合并
POSTOP_VARS_T3_COMBINED = list(set(POSTOP_VARS_T1 + POSTOP_VARS_T3))

VARIABLE_DESCRIPTIONS = {
    "us_well_defined_margin_preop": "US 边界清晰",
    "Etiology": "病因",
    "us_enlarged_lymph_nodes_preop": "US 淋巴结增大",
    "DWI_High_preop": "DWI 高信号",
    "Restricted_Diffusion_preop": "弥散受限",
    "Arterial_Enhancement_preop": "动脉期强化",
    "location": "肿瘤位置",
    "us_gallbladder_invasion_preop": "US 胆囊侵犯",
    "Margin_Ill_Defined_preop": "边界不清",
    "Number_of_lesions": "肿瘤数目",
    "Maximum_diameter": "肿瘤最大径 (cm)",

    "New_Nodule_post3m": "3月新发结节",
    "New_Nodule_Size_mm_post3m": "3月新结节大小 (mm)",
    "New_Nodule_Enhancement_post3m": "3月新结节强化",
    "New_Nodule_Count_post3m": "3月新结节数量",
    "Margil_Enhancement_post3m": "3月边缘强化",
    "Arterial_Enhancement_post3m": "3月动脉期强化",

    "Complete_Ablation_post6m": "6月完全消融",
    "Margil_Enhancement_post6m": "6月边缘强化",
    "New_Nodule_post6m": "6月新发结节",
    "New_Nodule_Count_post6m": "6月新结节数量",
    "New_Nodule_Size_mm_post6m": "6月新结节大小 (mm)",
    "New_Nodule_Enhancement_post6m": "6月新结节强化",
    "Arterial_Enhancement_post6m": "6月动脉期强化",

    "HSP70": "HSP70 (0-100)",
    "HIF_1α": "HIF-1α (0-100)",
    "BCL_2": "BCL-2 (0-100)",
    "MVI": "MVI",
    "E_cadherin": "E-cadherin (0-100)",
    "CK19": "CK19 (0-100)",
    "VEGF": "VEGF (0-100)",
    "MMP_9": "MMP-9 (0-100)",
}

# 二分类选项
BINARY_OPTIONS = ['(缺失)', '否', '是']
BINARY_ENCODING = {'否': 0, '是': 1}

CATEGORICAL_OPTIONS = {
    "us_well_defined_margin_preop": ['(缺失)', '否', '是'],
    "Etiology": ['(缺失)', 'Other', 'HBV', 'HCV', 'NAFLD', 'ALD'],
    "us_enlarged_lymph_nodes_preop": ['(缺失)', '否', '是'],
    "DWI_High_preop": ['(缺失)', '否', '是'],
    "Restricted_Diffusion_preop": ['(缺失)', '否', '是'],
    "Arterial_Enhancement_preop": ['(缺失)', '否', '是'],
    "location": ['(缺失)', 'Left', 'Right', 'Other'],
    "us_gallbladder_invasion_preop": ['(缺失)', '否', '是'],
    "Margin_Ill_Defined_preop": ['(缺失)', '否', '是'],

    "New_Nodule_post3m": ['(缺失)', '否', '是'],
    "New_Nodule_Enhancement_post3m": ['(缺失)', '否', '是'],
    "Margil_Enhancement_post3m": ['(缺失)', '否', '是'],
    "Arterial_Enhancement_post3m": ['(缺失)', '否', '是'],

    "Complete_Ablation_post6m": ['(缺失)', '否', '是'],
    "Margil_Enhancement_post6m": ['(缺失)', '否', '是'],
    "New_Nodule_post6m": ['(缺失)', '否', '是'],
    "New_Nodule_Enhancement_post6m": ['(缺失)', '否', '是'],
    "Arterial_Enhancement_post6m": ['(缺失)', '否', '是'],

    "MVI": ['(缺失)', '否', '是'],
}

CATEGORICAL_ENCODING = {
    "us_well_defined_margin_preop": {'否': 0, '是': 1},
    "Etiology": {'Other': 0, 'HBV': 1, 'HCV': 2, 'NAFLD': 3, 'ALD': 4},
    "us_enlarged_lymph_nodes_preop": {'否': 0, '是': 1},
    "DWI_High_preop": {'否': 0, '是': 1},
    "Restricted_Diffusion_preop": {'否': 0, '是': 1},
    "Arterial_Enhancement_preop": {'否': 0, '是': 1},
    "location": {'Left': 0, 'Right': 1, 'Other': 2},
    "us_gallbladder_invasion_preop": {'否': 0, '是': 1},
    "Margin_Ill_Defined_preop": {'否': 0, '是': 1},

    "New_Nodule_post3m": {'否': 0, '是': 1},
    "New_Nodule_Enhancement_post3m": {'否': 0, '是': 1},
    "Margil_Enhancement_post3m": {'否': 0, '是': 1},
    "Arterial_Enhancement_post3m": {'否': 0, '是': 1},

    "Complete_Ablation_post6m": {'否': 0, '是': 1},
    "Margil_Enhancement_post6m": {'否': 0, '是': 1},
    "New_Nodule_post6m": {'否': 0, '是': 1},
    "New_Nodule_Enhancement_post6m": {'否': 0, '是': 1},
    "Arterial_Enhancement_post6m": {'否': 0, '是': 1},

    "MVI": {'否': 0, '是': 1},
}


# ============================================================================
# 模型加载
# ============================================================================

@st.cache_resource
def load_models():
    """
    加载 T0 / T1 / T3 三个时点的模型 + scaler
    文件名：
      LIGHTCURE_Dynamic_T0.pth
      LIGHTCURE_Dynamic_T1.pth
      LIGHTCURE_Dynamic_T3.pth
      LIGHTCURE_Dynamic_T0_scaler.pkl
      LIGHTCURE_Dynamic_T1_scaler.pkl
      LIGHTCURE_Dynamic_T3_scaler.pkl
    """
    MODEL_DIRS = [
        'models',
        '../models',
        '.',
        r'D:/浙一/Papers/IRE预测模型/最终分析数据',
    ]

    def find_file(fname):
        for d in MODEL_DIRS:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
        return None

    def load_model_at_tp(tp_name, input_dim, preop_dim, postop_dim, ihc_dim):
        model_path = find_file(f'LIGHTCURE_Dynamic_{tp_name}.pth')
        scaler_path = find_file(f'LIGHTCURE_Dynamic_{tp_name}_scaler.pkl')

        if model_path is None:
            return None, None, f"模型文件缺失: LIGHTCURE_Dynamic_{tp_name}.pth"
        if scaler_path is None:
            return None, None, f"Scaler 缺失: LIGHTCURE_Dynamic_{tp_name}_scaler.pkl"

        try:
            model = LIGHTCURE_Dynamic_Model(
                input_dim=input_dim,
                preop_dim=preop_dim,
                postop_dim=postop_dim,
                ihc_dim=ihc_dim,
                hidden_dims=[192, 96, 48],
                dropout=0.45
            )
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            model.load_state_dict(state_dict)
            model.eval()

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            return model, scaler, None
        except Exception as e:
            return None, None, f"加载失败: {e}"

    # ---- T0 ----
    preop_dim = len(PREOP_MANDATORY_VARS)
    ihc_dim = len(IHC_ENHANCEMENT_VARS)

    # T0: 术前 + IHC, postop_dim = 0
    t0_input_dim = preop_dim + 0 + ihc_dim
    t0_model, t0_scaler, t0_err = load_model_at_tp(
        'T0', t0_input_dim, preop_dim, 0, ihc_dim
    )

    # T1: 术前 + 6 个术后 + IHC
    t1_postop_dim = len(POSTOP_VARS_T1)
    t1_input_dim = preop_dim + t1_postop_dim + ihc_dim
    t1_model, t1_scaler, t1_err = load_model_at_tp(
        'T1', t1_input_dim, preop_dim, t1_postop_dim, ihc_dim
    )

    # T3: 术前 + 13 个术后（T1+T3 合并）+ IHC
    t3_postop_dim = len(POSTOP_VARS_T3_COMBINED)
    t3_input_dim = preop_dim + t3_postop_dim + ihc_dim
    t3_model, t3_scaler, t3_err = load_model_at_tp(
        'T3', t3_input_dim, preop_dim, t3_postop_dim, ihc_dim
    )

    model_bundle = {
        'T0': {'model': t0_model, 'scaler': t0_scaler, 'postop_dim': 0,
               'postop_vars': [], 'err': t0_err},
        'T1': {'model': t1_model, 'scaler': t1_scaler, 'postop_dim': t1_postop_dim,
               'postop_vars': POSTOP_VARS_T1, 'err': t1_err},
        'T3': {'model': t3_model, 'scaler': t3_scaler, 'postop_dim': t3_postop_dim,
               'postop_vars': POSTOP_VARS_T3_COMBINED, 'err': t3_err},
    }

    # 状态提示
    for tp, d in model_bundle.items():
        if d['model'] is None:
            st.warning(f"⚠️ {tp} 模型未加载: {d['err']}")
        else:
            st.success(f"✅ {tp} 模型加载成功")

    return model_bundle


# ============================================================================
# 预测函数（支持 T0 / T1 / T3 三个时点）
# ============================================================================

def predict_at_timepoint(model_bundle, tp_name, input_dict):
    """
    tp_name: 'T0' / 'T1' / 'T3'
    input_dict: 用户输入的原始值（类别已编码为 int，缺失为 None）
    """
    bundle = model_bundle.get(tp_name)
    if bundle is None or bundle['model'] is None:
        return None

    model = bundle['model']
    scaler = bundle['scaler']
    postop_vars = bundle['postop_vars']

    # ---- 提取术前变量 ----
    X_preop = np.array([input_dict.get(v, 0) for v in PREOP_MANDATORY_VARS]).reshape(1, -1)

    # ---- 提取术后变量 ----
    X_postop = np.zeros((1, len(postop_vars)))
    postop_mask = np.zeros((1, len(postop_vars)))
    for i, v in enumerate(postop_vars):
        val = input_dict.get(v, None)
        if val is not None and not pd.isna(val):
            X_postop[0, i] = val
            postop_mask[0, i] = 1.0

    # ---- 提取 IHC 变量 ----
    X_ihc = np.zeros((1, len(IHC_ENHANCEMENT_VARS)))
    ihc_mask = np.zeros((1, len(IHC_ENHANCEMENT_VARS)))
    for i, v in enumerate(IHC_ENHANCEMENT_VARS):
        val = input_dict.get(v, None)
        if val is not None and not pd.isna(val):
            X_ihc[0, i] = val
            ihc_mask[0, i] = 1.0

    # ---- 合并原始特征 ----
    X_raw = np.concatenate([X_preop, X_postop, X_ihc], axis=1)

    # ---- 标准化（scaler 是按完整维度拟合的）----
    X_scaled = scaler.transform(X_raw)

    # ---- 送入模型 ----
    X_tensor = torch.FloatTensor(X_scaled)
    pm_tensor = torch.FloatTensor(postop_mask)
    im_tensor = torch.FloatTensor(ihc_mask)

    with torch.no_grad():
        prob = model(X_tensor, pm_tensor, im_tensor).numpy().flatten()

    prob = float(np.clip(prob[0], 1e-6, 1 - 1e-6))

    # ---- 风险分层（与主模型一致）----
    if prob < 0.2:
        risk_group = 'Low'
    elif prob < 0.5:
        risk_group = 'Intermediate'
    else:
        risk_group = 'High'

    # ---- 数据完整度 ----
    n_postop = int(postop_mask.sum())
    n_ihc = int(ihc_mask.sum())
    total_postop = len(postop_vars)
    total_ihc = len(IHC_ENHANCEMENT_VARS)
    total_vars = total_postop + total_ihc
    available_vars = n_postop + n_ihc
    completeness = (available_vars / total_vars * 100) if total_vars > 0 else 100.0

    # ---- IHC 热耐受表型 ----
    heat_present = []
    heat_sum = 0
    for m in ['HSP70', 'HIF_1α', 'BCL_2']:
        val = input_dict.get(m, None)
        if val is not None and not pd.isna(val):
            heat_present.append(m)
            heat_sum += val
    if len(heat_present) >= 3:
        heat_avg = heat_sum / 3
        heat_phenotype = 'Positive (High)' if heat_avg > 50 else 'Negative (Low)'
    elif len(heat_present) > 0:
        heat_avg = heat_sum / len(heat_present)
        heat_phenotype = f'Partial (n={len(heat_present)})'
    else:
        heat_avg = None
        heat_phenotype = 'Not Available'

    return {
        'P_LTP': prob,
        'risk_group': risk_group,
        'completeness': completeness,
        'n_postop': n_postop,
        'total_postop': total_postop,
        'n_ihc': n_ihc,
        'total_ihc': total_ihc,
        'heat_phenotype': heat_phenotype,
        'heat_markers_present': heat_present,
        'heat_score_avg': heat_avg,
    }


# ============================================================================
# 输入表单
# ============================================================================

def render_input_form():
    st.markdown("## 📋 Patient Information")
    st.caption("术前 11 变量 + 术后变量 + IHC 变量，全部允许缺失（缺失值走缺失嵌入）")

    tab1, tab2, tab3 = st.tabs(["🟥 术前 (必填)", "🟩 术后 (可选)", "🟪 IHC (可选)"])

    input_dict = {}

    # ---------------- 术前 ----------------
    with tab1:
        st.markdown("### 🟥 术前变量（11 个）")
        col1, col2 = st.columns(2)

        with col1:
            max_d = st.number_input("Maximum Diameter (cm)",
                                    min_value=0.1, max_value=15.0,
                                    value=2.0, step=0.1)
            input_dict['Maximum_diameter'] = max_d

            n_lesion = st.number_input("Number of Lesions",
                                       min_value=1, max_value=20, value=1)
            input_dict['Number_of_lesions'] = n_lesion

            etiology = st.selectbox("Etiology",
                                    options=CATEGORICAL_OPTIONS['Etiology'],
                                    index=0)
            if etiology != '(缺失)':
                input_dict['Etiology'] = CATEGORICAL_ENCODING['Etiology'][etiology]

            location = st.selectbox("Location",
                                    options=CATEGORICAL_OPTIONS['location'],
                                    index=0)
            if location != '(缺失)':
                input_dict['location'] = CATEGORICAL_ENCODING['location'][location]

            margin_ill = st.selectbox("Margin Ill-defined",
                                      options=CATEGORICAL_OPTIONS['Margin_Ill_Defined_preop'],
                                      index=0)
            if margin_ill != '(缺失)':
                input_dict['Margin_Ill_Defined_preop'] = CATEGORICAL_ENCODING['Margin_Ill_Defined_preop'][margin_ill]

        with col2:
            us_margin = st.selectbox("US Well-defined Margin",
                                     options=CATEGORICAL_OPTIONS['us_well_defined_margin_preop'],
                                     index=0)
            if us_margin != '(缺失)':
                input_dict['us_well_defined_margin_preop'] = CATEGORICAL_ENCODING['us_well_defined_margin_preop'][us_margin]

            dwi = st.selectbox("DWI High",
                               options=CATEGORICAL_OPTIONS['DWI_High_preop'], index=0)
            if dwi != '(缺失)':
                input_dict['DWI_High_preop'] = CATEGORICAL_ENCODING['DWI_High_preop'][dwi]

            rest = st.selectbox("Restricted Diffusion",
                                options=CATEGORICAL_OPTIONS['Restricted_Diffusion_preop'], index=0)
            if rest != '(缺失)':
                input_dict['Restricted_Diffusion_preop'] = CATEGORICAL_ENCODING['Restricted_Diffusion_preop'][rest]

            art = st.selectbox("Arterial Enhancement",
                               options=CATEGORICAL_OPTIONS['Arterial_Enhancement_preop'], index=0)
            if art != '(缺失)':
                input_dict['Arterial_Enhancement_preop'] = CATEGORICAL_ENCODING['Arterial_Enhancement_preop'][art]

            ln = st.selectbox("US Enlarged LN",
                              options=CATEGORICAL_OPTIONS['us_enlarged_lymph_nodes_preop'], index=0)
            if ln != '(缺失)':
                input_dict['us_enlarged_lymph_nodes_preop'] = CATEGORICAL_ENCODING['us_enlarged_lymph_nodes_preop'][ln]

            gb = st.selectbox("US Gallbladder Invasion",
                              options=CATEGORICAL_OPTIONS['us_gallbladder_invasion_preop'], index=0)
            if gb != '(缺失)':
                input_dict['us_gallbladder_invasion_preop'] = CATEGORICAL_ENCODING['us_gallbladder_invasion_preop'][gb]

    # ---------------- 术后 ----------------
    with tab2:
        st.markdown("### 🟩 术后变量（可选，全部允许缺失）")
        st.caption("术后 3 月变量 + 术后 6 月变量")

        st.markdown("**--- 术后 3 月变量 ---**")
        col1, col2 = st.columns(2)
        with col1:
            v = st.selectbox("New Nodule @3m",
                             options=CATEGORICAL_OPTIONS['New_Nodule_post3m'], index=0)
            if v != '(缺失)':
                input_dict['New_Nodule_post3m'] = CATEGORICAL_ENCODING['New_Nodule_post3m'][v]

            v = st.number_input("New Nodule Size @3m (mm)",
                                min_value=0.0, max_value=200.0,
                                value=None, step=1.0)
            if v is not None:
                input_dict['New_Nodule_Size_mm_post3m'] = v

            v = st.selectbox("New Nodule Enhancement @3m",
                             options=CATEGORICAL_OPTIONS['New_Nodule_Enhancement_post3m'], index=0)
            if v != '(缺失)':
                input_dict['New_Nodule_Enhancement_post3m'] = CATEGORICAL_ENCODING['New_Nodule_Enhancement_post3m'][v]

        with col2:
            v = st.number_input("New Nodule Count @3m",
                                min_value=0, max_value=50, value=None, step=1)
            if v is not None:
                input_dict['New_Nodule_Count_post3m'] = v

            v = st.selectbox("Marginal Enhancement @3m",
                             options=CATEGORICAL_OPTIONS['Margil_Enhancement_post3m'], index=0)
            if v != '(缺失)':
                input_dict['Margil_Enhancement_post3m'] = CATEGORICAL_ENCODING['Margil_Enhancement_post3m'][v]

            v = st.selectbox("Arterial Enhancement @3m",
                             options=CATEGORICAL_OPTIONS['Arterial_Enhancement_post3m'], index=0)
            if v != '(缺失)':
                input_dict['Arterial_Enhancement_post3m'] = CATEGORICAL_ENCODING['Arterial_Enhancement_post3m'][v]

        st.markdown("**--- 术后 6 月变量 ---**")
        col1, col2 = st.columns(2)
        with col1:
            v = st.selectbox("Complete Ablation @6m",
                             options=CATEGORICAL_OPTIONS['Complete_Ablation_post6m'], index=0)
            if v != '(缺失)':
                input_dict['Complete_Ablation_post6m'] = CATEGORICAL_ENCODING['Complete_Ablation_post6m'][v]

            v = st.selectbox("Marginal Enhancement @6m",
                             options=CATEGORICAL_OPTIONS['Margil_Enhancement_post6m'], index=0)
            if v != '(缺失)':
                input_dict['Margil_Enhancement_post6m'] = CATEGORICAL_ENCODING['Margil_Enhancement_post6m'][v]

            v = st.selectbox("New Nodule @6m",
                             options=CATEGORICAL_OPTIONS['New_Nodule_post6m'], index=0)
            if v != '(缺失)':
                input_dict['New_Nodule_post6m'] = CATEGORICAL_ENCODING['New_Nodule_post6m'][v]

            v = st.number_input("New Nodule Count @6m",
                                min_value=0, max_value=50, value=None, step=1)
            if v is not None:
                input_dict['New_Nodule_Count_post6m'] = v

        with col2:
            v = st.number_input("New Nodule Size @6m (mm)",
                                min_value=0.0, max_value=200.0,
                                value=None, step=1.0)
            if v is not None:
                input_dict['New_Nodule_Size_mm_post6m'] = v

            v = st.selectbox("New Nodule Enhancement @6m",
                             options=CATEGORICAL_OPTIONS['New_Nodule_Enhancement_post6m'], index=0)
            if v != '(缺失)':
                input_dict['New_Nodule_Enhancement_post6m'] = CATEGORICAL_ENCODING['New_Nodule_Enhancement_post6m'][v]

            v = st.selectbox("Arterial Enhancement @6m",
                             options=CATEGORICAL_OPTIONS['Arterial_Enhancement_post6m'], index=0)
            if v != '(缺失)':
                input_dict['Arterial_Enhancement_post6m'] = CATEGORICAL_ENCODING['Arterial_Enhancement_post6m'][v]

    # ---------------- IHC ----------------
    with tab3:
        st.markdown("### 🟪 IHC 变量（可选，全部允许缺失）")
        col1, col2 = st.columns(2)
        with col1:
            for v in ['HSP70', 'HIF_1α', 'BCL_2', 'E_cadherin']:
                val = st.number_input(VARIABLE_DESCRIPTIONS[v],
                                      min_value=0.0, max_value=100.0,
                                      value=None, step=1.0, key=f'ihc_{v}')
                if val is not None:
                    input_dict[v] = val

        with col2:
            for v in ['CK19', 'VEGF', 'MMP_9']:
                val = st.number_input(VARIABLE_DESCRIPTIONS[v],
                                      min_value=0.0, max_value=100.0,
                                      value=None, step=1.0, key=f'ihc_{v}')
                if val is not None:
                    input_dict[v] = val

            mvi = st.selectbox("MVI", options=CATEGORICAL_OPTIONS['MVI'], index=0, key='mvi_sel')
            if mvi != '(缺失)':
                input_dict['MVI'] = CATEGORICAL_ENCODING['MVI'][mvi]

    return input_dict


# ============================================================================
# 结果渲染
# ============================================================================

def render_timepoint_result(tp_name, res):
    if res is None:
        st.warning(f"{tp_name} 模型未加载，无法预测")
        return

    st.markdown(f"### {tp_name} 时点预测结果")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted LTP Probability", f"{res['P_LTP']*100:.1f}%")
    with col2:
        rg = res['risk_group']
        if rg == 'Low':
            st.markdown('<span class="risk-low">🟢 Low Risk</span>', unsafe_allow_html=True)
        elif rg == 'Intermediate':
            st.markdown('<span class="risk-intermediate">🟡 Intermediate</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-high">🔴 High Risk</span>', unsafe_allow_html=True)
    with col3:
        st.metric("Completeness", f"{res['completeness']:.0f}%")

    if res['heat_phenotype'] != 'Not Available':
        st.markdown(f'<span class="heat-phenotype">🔥 Heat Phenotype: {res["heat_phenotype"]}</span>',
                    unsafe_allow_html=True)
        if res['heat_markers_present']:
            st.caption(f"Markers: {', '.join(res['heat_markers_present'])} "
                       f"(Avg: {res['heat_score_avg']:.1f})")


def render_results(model_bundle, input_dict):
    st.markdown("## 📊 Prediction Results")

    # 判断可用的时点
    t0 = predict_at_timepoint(model_bundle, 'T0', input_dict)
    t1 = predict_at_timepoint(model_bundle, 'T1', input_dict)
    t3 = predict_at_timepoint(model_bundle, 'T3', input_dict)

    tab0, tab1, tab3_ = st.tabs(["T0 (术前)", "T1 (术后 3 月)", "T3 (术后 6 月)"])

    with tab0:
        render_timepoint_result('T0', t0)
    with tab1:
        render_timepoint_result('T1', t1)
    with tab3_:
        render_timepoint_result('T3', t3)

    st.markdown("---")
    st.markdown("### 时点对比")

    rows = []
    for tp, r in [('T0 (术前)', t0), ('T1 (术后 3 月)', t1), ('T3 (术后 6 月)', t3)]:
        if r is not None:
            rows.append({
                'Timepoint': tp,
                'Predicted LTP': f"{r['P_LTP']*100:.1f}%",
                'Risk Group': r['risk_group'],
                'Completeness': f"{r['completeness']:.0f}%",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ============================================================================
# 主程序
# ============================================================================

def main():
    st.markdown('<div class="main-header">🏥 LIGHTCURE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Postoperative Dynamic Monitoring System<br>'
        '<small>术后动态监测系统</small>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        **LIGHTCURE-Dynamic** 是一个基于深度学习的术后动态监测系统。

        **核心设计：**
        - **T0**：术前 11 变量 + IHC
        - **T1**：术前 + 术后 3 月变量 + IHC
        - **T3**：术前 + 术后 3 月 + 6 月变量 + IHC

        **特性：**
        - IHC 和术后变量都允许缺失（走缺失嵌入）
        - 缺失越多，预测置信度越低
        - 自动显示热耐受表型（HSP70/HIF-1α/BCL-2）

        **隐私：**
        - 所有计算在本地进行
        - 不存储患者数据
        """)
        st.markdown("---")

    if 'model_bundle' not in st.session_state:
        with st.spinner("⏳ 加载模型中..."):
            bundle = load_models()
            st.session_state.model_bundle = bundle
            st.session_state.models_loaded = True

    input_dict = render_input_form()

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        try:
            render_results(st.session_state.model_bundle, input_dict)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)

    st.markdown("---")
    st.markdown("""
    <div class="footer">
        LIGHTCURE-Dynamic v4.0 | For clinical research use only | No patient data stored
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()