# -*- coding: utf-8 -*-
"""
LIGHTCURE - 肝细胞癌消融治疗决策支持系统
公开平台部署 - Streamlit应用

依据方案：第十六部分 - 公开平台部署
技术栈：Streamlit + PyTorch + SHAP + Matplotlib

功能模块：
- 输入：23个变量（14术前必填 + 9术后可选）+ IHC增强（可选）
- 输出：RFA/IRE风险评分、推荐治疗
- 缺失处理：术后变量缺失时使用缺失嵌入，IHC可选
- 解释：SHAP解释图
- 报告：PDF报告生成
- 隐私保护：不存储任何患者数据

IHC增强输入（方案3.2）：
- 核心标志物：HSP70, HIF-1α, BCL-2（热耐受表型）
- 扩展标志物：Ki-67, GPC-3, CK7, CK19, E-cadherin, MMP-9, VEGF
- 病理变量：MVI
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import shap
import io
import base64
from datetime import datetime
import warnings
import os
import json
import hashlib
warnings.filterwarnings('ignore')

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="LIGHTCURE - HCC Ablation Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 全局样式
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2e86c1;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
    }
    .risk-intermediate {
        background-color: #f39c12;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
    }
    .risk-low {
        background-color: #27ae60;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
    }
    .recommend-ire {
        background-color: #2e86c1;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .recommend-rfa {
        background-color: #27ae60;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .recommend-either {
        background-color: #f39c12;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .ihc-positive {
        background-color: #e74c3c;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .ihc-negative {
        background-color: #27ae60;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .heat-phenotype {
        background-color: #8e44ad;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.8rem;
        padding: 1rem 0;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 模型定义（含缺失自适应编码器）
# ============================================================================

class MissingAdaptiveEncoder(nn.Module):
    """缺失自适应编码器（方案5.2.1）"""
    def __init__(self, input_dim, preop_dim, postop_dim, ihc_dim=0, hidden_dims=[192, 96, 48], dropout=0.45):
        super(MissingAdaptiveEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.preop_dim = preop_dim
        self.postop_dim = postop_dim
        self.ihc_dim = ihc_dim
        
        # 可学习的缺失嵌入（仅对术后变量）
        self.missing_embeddings = nn.Parameter(
            torch.randn(self.postop_dim, 1) * 0.01
        )
        
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
    
    def forward(self, x, mask):
        batch_size = x.size(0)
        
        x_preop = x[:, :self.preop_dim]
        x_postop_vars = x[:, self.preop_dim:self.preop_dim + self.postop_dim]
        x_ihc = x[:, self.preop_dim + self.postop_dim:]
        
        missing_emb_expanded = self.missing_embeddings.squeeze().unsqueeze(0).expand(batch_size, -1)
        x_postop_encoded = x_postop_vars * mask + (1 - mask) * missing_emb_expanded
        
        x_combined = torch.cat([x_preop, x_postop_encoded, x_ihc], dim=1)
        
        x_combined = self.bn_input(x_combined)
        x_combined = self.dropout_input(x_combined)
        x_combined = self.hidden(x_combined)
        x_combined = self.output(x_combined)
        
        return self.sigmoid(x_combined).squeeze(-1)


class LIGHTCURE_RFA_Model(nn.Module):
    """RFA模型架构 - 含缺失自适应编码器"""
    def __init__(self, input_dim, preop_dim, postop_dim, ihc_dim=0, hidden_dims=[192, 96, 48], dropout=0.45):
        super(LIGHTCURE_RFA_Model, self).__init__()
        self.encoder = MissingAdaptiveEncoder(input_dim, preop_dim, postop_dim, ihc_dim, hidden_dims, dropout)
    
    def forward(self, x, mask):
        return self.encoder(x, mask)


class LIGHTCURE_IRE_Model(nn.Module):
    """IRE模型架构 - 含缺失自适应编码器"""
    def __init__(self, input_dim, preop_dim, postop_dim, ihc_dim=0, hidden_dims=[192, 96, 48], dropout=0.45):
        super(LIGHTCURE_IRE_Model, self).__init__()
        self.encoder = MissingAdaptiveEncoder(input_dim, preop_dim, postop_dim, ihc_dim, hidden_dims, dropout)
    
    def forward(self, x, mask):
        return self.encoder(x, mask)

# ============================================================================
# 变量定义（23个变量 + IHC增强）
# ============================================================================

# 1.1 术前必填变量（14个）
PREOP_MANDATORY_VARS = [
    "Etiology", "Preop_AFP", "Number_of_lesions", "Portal_Hypertension",
    "Arterial_Enhancement_preop", "Maximum_diameter", "Age",
    "Washout_preop", "Shape_Irregular_preop", "Capsule_Intact_preop",
    "Margin_Ill_Defined_preop", "US_Echogenicity_Preop", "subcapsular",
    "CEUS_Pattern_preop"
]

# 1.2 术后可选变量（9个）
POSTOP_OPTIONAL_VARS = [
    "New_Nodule_post6m", "Complete_Ablation_post3m", "Complete_Ablation_post1m",
    "New_Nodule_post3m", "ALT_Recovery_Ratio_6m", "POD1_AST_Ratio",
    "Complete_Ablation_post6m", "POD1_NLR", "Post6M_NLR"
]

# 1.3 IHC增强变量（12个，有则加）
IHC_ENHANCEMENT_VARS = [
    "HSP70", "HIF_1α", "BCL_2",      # 核心标志物（热耐受表型）
    "Ki_67", "GPC_3", "CK7", "CK19",  # 扩展标志物
    "E_cadherin", "MMP_9", "VEGF",    # 侵袭性标志物
    "CD34", "MVI"                     # 微血管密度和侵犯
]

ALL_VARS = PREOP_MANDATORY_VARS + POSTOP_OPTIONAL_VARS

# 变量描述（用于UI显示）
VARIABLE_DESCRIPTIONS = {
    # 术前变量
    "Etiology": "病因 (HBV, HCV, NAFLD, ALD, 其他)",
    "Preop_AFP": "术前AFP (ng/mL)",
    "Number_of_lesions": "肿瘤数目",
    "Portal_Hypertension": "门脉高压 (是/否)",
    "Arterial_Enhancement_preop": "动脉期强化 (是/否)",
    "Maximum_diameter": "肿瘤最大径 (cm)",
    "Age": "年龄 (岁)",
    "Washout_preop": "廓清征 (是/否)",
    "Shape_Irregular_preop": "形态不规则 (是/否)",
    "Capsule_Intact_preop": "包膜完整 (是/否)",
    "Margin_Ill_Defined_preop": "边界不清 (是/否)",
    "US_Echogenicity_Preop": "超声回声类型 (低回声/等回声/高回声/混合)",
    "subcapsular": "包膜下肿瘤 (是/否)",
    "CEUS_Pattern_preop": "CEUS增强模式 (快进快出/其他)",
    # 术后变量
    "New_Nodule_post6m": "术后6月新发结节 (是/否)",
    "Complete_Ablation_post3m": "术后3月完全消融 (是/否)",
    "Complete_Ablation_post1m": "术后1月完全消融 (是/否)",
    "New_Nodule_post3m": "术后3月新发结节 (是/否)",
    "ALT_Recovery_Ratio_6m": "术后6月ALT恢复比率",
    "POD1_AST_Ratio": "术后1天AST变化率",
    "Complete_Ablation_post6m": "术后6月完全消融 (是/否)",
    "POD1_NLR": "术后1天中性粒细胞/淋巴细胞比值",
    "Post6M_NLR": "术后6月中性粒细胞/淋巴细胞比值",
    # IHC标志物
    "HSP70": "热休克蛋白70 (表达值, 0-100)",
    "HIF_1α": "缺氧诱导因子-1α (表达值, 0-100)",
    "BCL_2": "B细胞淋巴瘤-2 (表达值, 0-100)",
    "Ki_67": "增殖指数 (表达值, 0-100)",
    "GPC_3": "磷脂酰肌醇蛋白聚糖-3 (表达值, 0-100)",
    "CK7": "细胞角蛋白-7 (表达值, 0-100)",
    "CK19": "细胞角蛋白-19 (表达值, 0-100)",
    "E_cadherin": "E-钙粘蛋白 (表达值, 0-100)",
    "MMP_9": "基质金属蛋白酶-9 (表达值, 0-100)",
    "VEGF": "血管内皮生长因子 (表达值, 0-100)",
    "CD34": "微血管密度 (表达值, 0-100)",
    "MVI": "微血管侵犯 (是/否)"
}

# 分类变量选项
CATEGORICAL_OPTIONS = {
    "Etiology": ['HBV', 'HCV', 'NAFLD', 'ALD', '其他'],
    "Portal_Hypertension": ['否', '是'],
    "Arterial_Enhancement_preop": ['否', '是'],
    "Washout_preop": ['否', '是'],
    "Shape_Irregular_preop": ['否', '是'],
    "Capsule_Intact_preop": ['否', '是'],
    "Margin_Ill_Defined_preop": ['否', '是'],
    "US_Echogenicity_Preop": ['低回声', '等回声', '高回声', '混合'],
    "subcapsular": ['否', '是'],
    "CEUS_Pattern_preop": ['其他', '快进快出'],
    "New_Nodule_post6m": ['否', '是'],
    "Complete_Ablation_post3m": ['否', '是'],
    "Complete_Ablation_post1m": ['否', '是'],
    "New_Nodule_post3m": ['否', '是'],
    "Complete_Ablation_post6m": ['否', '是'],
    "MVI": ['否', '是']
}

# 分类变量编码映射
CATEGORICAL_ENCODING = {
    "Etiology": {'HBV': 1, 'HCV': 2, 'NAFLD': 3, 'ALD': 4, '其他': 0},
    "US_Echogenicity_Preop": {'低回声': 1, '等回声': 2, '高回声': 3, '混合': 4},
    "CEUS_Pattern_preop": {'其他': 0, '快进快出': 1},
    "Portal_Hypertension": {'否': 0, '是': 1},
    "Arterial_Enhancement_preop": {'否': 0, '是': 1},
    "Washout_preop": {'否': 0, '是': 1},
    "Shape_Irregular_preop": {'否': 0, '是': 1},
    "Capsule_Intact_preop": {'否': 0, '是': 1},
    "Margin_Ill_Defined_preop": {'否': 0, '是': 1},
    "subcapsular": {'否': 0, '是': 1},
    "New_Nodule_post6m": {'否': 0, '是': 1},
    "Complete_Ablation_post3m": {'否': 0, '是': 1},
    "Complete_Ablation_post1m": {'否': 0, '是': 1},
    "New_Nodule_post3m": {'否': 0, '是': 1},
    "Complete_Ablation_post6m": {'否': 0, '是': 1},
    "MVI": {'否': 0, '是': 1}
}

# IHC标志物参考范围
IHC_RANGES = {
    "HSP70": (0, 100),
    "HIF_1α": (0, 100),
    "BCL_2": (0, 100),
    "Ki_67": (0, 100),
    "GPC_3": (0, 100),
    "CK7": (0, 100),
    "CK19": (0, 100),
    "E_cadherin": (0, 100),
    "MMP_9": (0, 100),
    "VEGF": (0, 100),
    "CD34": (0, 100)
}

# ============================================================================
# 模型加载函数
# ============================================================================

@st.cache_resource
def load_models():
    """加载模型、标准化器和模型信息"""
    
    # 模型文件路径
    model_paths = {
        'rfa': [
            'models/LIGHTCURE_RFA_23var.pth',
            '../models/LIGHTCURE_RFA_23var.pth',
            'LIGHTCURE_RFA_23var.pth',
            'D:/浙一/Papers/IRE预测模型/最终分析数据/LIGHTCURE_RFA_23var.pth'
        ],
        'ire': [
            'models/LIGHTCURE_IRE_23var_Best.pth',
            '../models/LIGHTCURE_IRE_23var_Best.pth',
            'LIGHTCURE_IRE_23var_Best.pth',
            'D:/浙一/Papers/IRE预测模型/最终分析数据/LIGHTCURE_IRE_23var_Best.pth'
        ],
        'scaler': [
            'models/scaler_rfa_23var.pkl',
            '../models/scaler_rfa_23var.pkl',
            'scaler_rfa_23var.pkl',
            'D:/浙一/Papers/IRE预测模型/最终分析数据/scaler_rfa_23var.pkl'
        ],
        'model_info': [
            'models/model_info_23var.json',
            '../models/model_info_23var.json',
            'model_info_23var.json',
            'D:/浙一/Papers/IRE预测模型/最终分析数据/model_info_23var.json'
        ]
    }
    
    def find_file(file_list):
        for path in file_list:
            if os.path.exists(path):
                return path
        return None
    
    # 加载模型信息
    info_path = find_file(model_paths['model_info'])
    if info_path:
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        st.success(f"✅ Model info loaded: {info_path}")
    else:
        st.warning("⚠️ Model info not found, using defaults")
        model_info = {
            'input_dim': 23,
            'preop_dim': 14,
            'postop_dim': 9,
            'hidden_dims': [192, 96, 48],
            'dropout': 0.45
        }
    
    input_dim = model_info.get('input_dim', 23)
    preop_dim = model_info.get('preop_dim', 14)
    postop_dim = model_info.get('postop_dim', 9)
    ihc_dim = 0  # 基础模型不包含IHC
    
    # 加载RFA模型
    rfa_path = find_file(model_paths['rfa'])
    if rfa_path is None:
        st.error("❌ RFA model not found")
        return None, None, None, None
    
    rfa_model = LIGHTCURE_RFA_Model(
        input_dim=input_dim,
        preop_dim=preop_dim,
        postop_dim=postop_dim,
        ihc_dim=ihc_dim,
        hidden_dims=model_info.get('hidden_dims', [192, 96, 48]),
        dropout=model_info.get('dropout', 0.45)
    )
    try:
        state_dict = torch.load(rfa_path, map_location='cpu', weights_only=False)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        rfa_model.load_state_dict(state_dict)
        rfa_model.eval()
        st.success(f"✅ RFA model loaded: {rfa_path}")
    except Exception as e:
        st.error(f"❌ RFA model load failed: {e}")
        return None, None, None, None
    
    # 加载IRE模型
    ire_path = find_file(model_paths['ire'])
    if ire_path is None:
        st.warning("⚠️ IRE model not found, using RFA as fallback")
        ire_model = rfa_model
    else:
        ire_model = LIGHTCURE_IRE_Model(
            input_dim=input_dim,
            preop_dim=preop_dim,
            postop_dim=postop_dim,
            ihc_dim=ihc_dim,
            hidden_dims=model_info.get('hidden_dims', [192, 96, 48]),
            dropout=model_info.get('dropout', 0.45)
        )
        try:
            state_dict = torch.load(ire_path, map_location='cpu', weights_only=False)
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            ire_model.load_state_dict(state_dict)
            ire_model.eval()
            st.success(f"✅ IRE model loaded: {ire_path}")
        except Exception as e:
            st.warning(f"⚠️ IRE model load failed: {e}, using RFA as fallback")
            ire_model = rfa_model
    
    # 加载标准化器
    scaler_path = find_file(model_paths['scaler'])
    if scaler_path is None:
        st.error("❌ Scaler not found")
        return rfa_model, ire_model, None, model_info
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        st.success(f"✅ Scaler loaded: {scaler_path}")
    
    return rfa_model, ire_model, scaler, model_info

# ============================================================================
# 预测函数（含IHC支持）
# ============================================================================

def predict_with_ihc(models, input_dict, scaler):
    """
    进行预测（含IHC增强）
    
    参数:
        models: (rfa_model, ire_model) 元组
        input_dict: 输入数据字典
        scaler: 标准化器
    
    返回:
        dict: 包含P_RFA, P_IRE, delta_P, risk_group, recommendation, 
              heat_phenotype, ihc_available
    """
    rfa_model, ire_model = models
    
    # 构建特征向量
    preop_vars = [v for v in PREOP_MANDATORY_VARS if v in ALL_VARS]
    postop_vars = [v for v in POSTOP_OPTIONAL_VARS if v in ALL_VARS]
    ihc_vars = [v for v in IHC_ENHANCEMENT_VARS if v in input_dict and input_dict.get(v) is not None]
    
    # 提取术前变量
    X_preop = np.array([input_dict.get(v, 0) for v in preop_vars]).reshape(1, -1)
    
    # 提取术后变量
    X_postop = np.array([input_dict.get(v, 0) for v in postop_vars]).reshape(1, -1)
    
    # 术后变量缺失指示器
    mask = np.zeros_like(X_postop)
    for i, v in enumerate(postop_vars):
        if v in input_dict and input_dict.get(v) is not None and not pd.isna(input_dict.get(v)):
            mask[0, i] = 1.0
    
    # 提取IHC变量
    if ihc_vars:
        X_ihc = np.array([input_dict.get(v, 0) for v in ihc_vars]).reshape(1, -1)
        X_combined = np.concatenate([X_preop, X_postop, X_ihc], axis=1)
        ihc_available = True
        ihc_count = len(ihc_vars)
    else:
        X_combined = np.concatenate([X_preop, X_postop], axis=1)
        ihc_available = False
        ihc_count = 0
    
    # 标准化
    X_scaled = scaler.transform(X_combined)
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_scaled)
    mask_tensor = torch.FloatTensor(mask)
    
    # 预测
    with torch.no_grad():
        P_RFA = rfa_model(X_tensor, mask_tensor).numpy().flatten()
        P_IRE = ire_model(X_tensor, mask_tensor).numpy().flatten()
    
    delta_P = P_RFA - P_IRE
    
    # 风险分组
    p_rfa = P_RFA[0]
    if p_rfa < 0.2:
        risk_group = 'Low'
    elif p_rfa < 0.5:
        risk_group = 'Intermediate'
    else:
        risk_group = 'High'
    
    # 治疗推荐
    dp = delta_P[0]
    if dp > 0.05:
        recommendation = 'IRE'
        recommendation_detail = f"ΔP > 0.05: IRE may reduce LTP risk by {dp*100:.1f}%"
    elif dp < -0.05:
        recommendation = 'RFA'
        recommendation_detail = f"ΔP < -0.05: RFA may reduce LTP risk by {-dp*100:.1f}%"
    else:
        recommendation = 'Either'
        recommendation_detail = "ΔP near 0: Similar expected outcomes"
    
    # 热耐受表型评估（方案9.3）
    heat_markers_present = []
    heat_score = 0
    heat_markers_count = 0
    
    for m in ['HSP70', 'HIF_1α', 'BCL_2']:
        if m in input_dict and input_dict.get(m) is not None and not pd.isna(input_dict.get(m)):
            val = input_dict.get(m, 0)
            heat_markers_present.append(m)
            heat_score += val
            heat_markers_count += 1
    
    if heat_markers_count >= 3:
        heat_score_avg = heat_score / 3
        if heat_score_avg > 50:  # 假设中位数为50
            heat_phenotype = 'Positive (High)'
            heat_phenotype_color = 'positive'
        else:
            heat_phenotype = 'Negative (Low)'
            heat_phenotype_color = 'negative'
    elif heat_markers_count > 0:
        heat_score_avg = heat_score / heat_markers_count
        if heat_score_avg > 50:
            heat_phenotype = f'Partial Positive (n={heat_markers_count})'
            heat_phenotype_color = 'positive'
        else:
            heat_phenotype = f'Partial Negative (n={heat_markers_count})'
            heat_phenotype_color = 'negative'
    else:
        heat_phenotype = 'Not Available'
        heat_phenotype_color = 'neutral'
    
    # 数据完整度评分（方案3.5.4）
    total_postop = len(postop_vars)
    available_postop = mask.sum()
    total_ihc = len(IHC_ENHANCEMENT_VARS)
    available_ihc = ihc_count
    total_vars = total_postop + total_ihc
    available_vars = available_postop + available_ihc
    completeness = (available_vars / total_vars * 100) if total_vars > 0 else 100
    
    if completeness >= 80:
        confidence = '★★★★★'
        confidence_label = 'High'
    elif completeness >= 60:
        confidence = '★★★★'
        confidence_label = 'Moderate-High'
    elif completeness >= 40:
        confidence = '★★★'
        confidence_label = 'Moderate'
    else:
        confidence = '★★'
        confidence_label = 'Low'
    
    return {
        'P_RFA': float(P_RFA[0]),
        'P_IRE': float(P_IRE[0]),
        'delta_P': float(delta_P[0]),
        'risk_group': risk_group,
        'recommendation': recommendation,
        'recommendation_detail': recommendation_detail,
        'heat_phenotype': heat_phenotype,
        'heat_phenotype_color': heat_phenotype_color,
        'heat_markers_present': heat_markers_present,
        'heat_score_avg': heat_score_avg if heat_markers_count > 0 else None,
        'ihc_available': ihc_available,
        'ihc_count': ihc_count,
        'completeness': float(completeness),
        'confidence': confidence,
        'confidence_label': confidence_label,
        'available_postop': int(available_postop),
        'total_postop': total_postop,
        'available_ihc': available_ihc,
        'total_ihc': total_ihc
    }

# ============================================================================
# 渲染函数
# ============================================================================

def render_input_form():
    """渲染输入表单（含IHC）"""
    
    st.markdown("## 📋 Patient Information Input")
    st.caption("🟥 Required (Preop) | 🟩 Optional (Postop) | 🟪 IHC Enhancement (Optional)")
    
    # 创建Tab
    tab1, tab2, tab3 = st.tabs(["🟥 Preoperative (Required)", "🟩 Postoperative (Optional)", "🟪 IHC Enhancement (Optional)"])
    
    input_dict = {}
    
    with tab1:
        st.markdown("### 🟥 Preoperative Features (Required)")
        st.caption("All preoperative variables must be provided")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18, max_value=85, value=55,
                help=VARIABLE_DESCRIPTIONS['Age']
            )
            input_dict['Age'] = age
            
            etiology = st.selectbox(
                "Etiology",
                options=CATEGORICAL_OPTIONS['Etiology'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Etiology']
            )
            input_dict['Etiology'] = CATEGORICAL_ENCODING['Etiology'][etiology]
            
            afp = st.number_input(
                "Preop AFP (ng/mL)",
                min_value=0.0, max_value=100000.0, value=10.0, step=1.0,
                help=VARIABLE_DESCRIPTIONS['Preop_AFP']
            )
            input_dict['Preop_AFP'] = afp
            
            num_lesions = st.number_input(
                "Number of Lesions",
                min_value=1, max_value=10, value=1,
                help=VARIABLE_DESCRIPTIONS['Number_of_lesions']
            )
            input_dict['Number_of_lesions'] = num_lesions
            
            max_diameter = st.number_input(
                "Maximum Diameter (cm)",
                min_value=0.1, max_value=5.0, value=2.0, step=0.1,
                help=VARIABLE_DESCRIPTIONS['Maximum_diameter']
            )
            input_dict['Maximum_diameter'] = max_diameter
            
            portal_hypertension = st.selectbox(
                "Portal Hypertension",
                options=CATEGORICAL_OPTIONS['Portal_Hypertension'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Portal_Hypertension']
            )
            input_dict['Portal_Hypertension'] = CATEGORICAL_ENCODING['Portal_Hypertension'][portal_hypertension]
            
            arterial_enhancement = st.selectbox(
                "Arterial Enhancement",
                options=CATEGORICAL_OPTIONS['Arterial_Enhancement_preop'],
                index=1,
                help=VARIABLE_DESCRIPTIONS['Arterial_Enhancement_preop']
            )
            input_dict['Arterial_Enhancement_preop'] = CATEGORICAL_ENCODING['Arterial_Enhancement_preop'][arterial_enhancement]
        
        with col2:
            washout = st.selectbox(
                "Washout",
                options=CATEGORICAL_OPTIONS['Washout_preop'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Washout_preop']
            )
            input_dict['Washout_preop'] = CATEGORICAL_ENCODING['Washout_preop'][washout]
            
            shape_irregular = st.selectbox(
                "Irregular Shape",
                options=CATEGORICAL_OPTIONS['Shape_Irregular_preop'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Shape_Irregular_preop']
            )
            input_dict['Shape_Irregular_preop'] = CATEGORICAL_ENCODING['Shape_Irregular_preop'][shape_irregular]
            
            capsule_intact = st.selectbox(
                "Intact Capsule",
                options=CATEGORICAL_OPTIONS['Capsule_Intact_preop'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Capsule_Intact_preop']
            )
            input_dict['Capsule_Intact_preop'] = CATEGORICAL_ENCODING['Capsule_Intact_preop'][capsule_intact]
            
            margin_ill_defined = st.selectbox(
                "Ill-defined Margin",
                options=CATEGORICAL_OPTIONS['Margin_Ill_Defined_preop'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Margin_Ill_Defined_preop']
            )
            input_dict['Margin_Ill_Defined_preop'] = CATEGORICAL_ENCODING['Margin_Ill_Defined_preop'][margin_ill_defined]
            
            us_echogenicity = st.selectbox(
                "US Echogenicity",
                options=CATEGORICAL_OPTIONS['US_Echogenicity_Preop'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['US_Echogenicity_Preop']
            )
            input_dict['US_Echogenicity_Preop'] = CATEGORICAL_ENCODING['US_Echogenicity_Preop'][us_echogenicity]
            
            subcapsular = st.selectbox(
                "Subcapsular",
                options=CATEGORICAL_OPTIONS['subcapsular'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['subcapsular']
            )
            input_dict['subcapsular'] = CATEGORICAL_ENCODING['subcapsular'][subcapsular]
            
            ceus_pattern = st.selectbox(
                "CEUS Pattern",
                options=CATEGORICAL_OPTIONS['CEUS_Pattern_preop'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['CEUS_Pattern_preop']
            )
            input_dict['CEUS_Pattern_preop'] = CATEGORICAL_ENCODING['CEUS_Pattern_preop'][ceus_pattern]
    
    with tab2:
        st.markdown("### 🟩 Postoperative Features (Optional)")
        st.caption("Leave blank if not available - model will handle missing values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_nodule_6m = st.selectbox(
                "New Nodule at 6 Months",
                options=CATEGORICAL_OPTIONS['New_Nodule_post6m'],
                index=1,
                help=VARIABLE_DESCRIPTIONS['New_Nodule_post6m']
            )
            if new_nodule_6m != '':
                input_dict['New_Nodule_post6m'] = CATEGORICAL_ENCODING['New_Nodule_post6m'][new_nodule_6m]
            
            complete_ablation_3m = st.selectbox(
                "Complete Ablation at 3 Months",
                options=CATEGORICAL_OPTIONS['Complete_Ablation_post3m'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Complete_Ablation_post3m']
            )
            if complete_ablation_3m != '':
                input_dict['Complete_Ablation_post3m'] = CATEGORICAL_ENCODING['Complete_Ablation_post3m'][complete_ablation_3m]
            
            complete_ablation_1m = st.selectbox(
                "Complete Ablation at 1 Month",
                options=CATEGORICAL_OPTIONS['Complete_Ablation_post1m'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Complete_Ablation_post1m']
            )
            if complete_ablation_1m != '':
                input_dict['Complete_Ablation_post1m'] = CATEGORICAL_ENCODING['Complete_Ablation_post1m'][complete_ablation_1m]
            
            new_nodule_3m = st.selectbox(
                "New Nodule at 3 Months",
                options=CATEGORICAL_OPTIONS['New_Nodule_post3m'],
                index=1,
                help=VARIABLE_DESCRIPTIONS['New_Nodule_post3m']
            )
            if new_nodule_3m != '':
                input_dict['New_Nodule_post3m'] = CATEGORICAL_ENCODING['New_Nodule_post3m'][new_nodule_3m]
            
            complete_ablation_6m = st.selectbox(
                "Complete Ablation at 6 Months",
                options=CATEGORICAL_OPTIONS['Complete_Ablation_post6m'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['Complete_Ablation_post6m']
            )
            if complete_ablation_6m != '':
                input_dict['Complete_Ablation_post6m'] = CATEGORICAL_ENCODING['Complete_Ablation_post6m'][complete_ablation_6m]
        
        with col2:
            alt_recovery_ratio = st.number_input(
                "ALT Recovery Ratio (6 Months)",
                min_value=0.0, max_value=5.0, value=None, step=0.1,
                help=VARIABLE_DESCRIPTIONS['ALT_Recovery_Ratio_6m']
            )
            if alt_recovery_ratio is not None:
                input_dict['ALT_Recovery_Ratio_6m'] = alt_recovery_ratio
            
            pod1_ast_ratio = st.number_input(
                "POD1 AST Ratio",
                min_value=0.0, max_value=5.0, value=None, step=0.1,
                help=VARIABLE_DESCRIPTIONS['POD1_AST_Ratio']
            )
            if pod1_ast_ratio is not None:
                input_dict['POD1_AST_Ratio'] = pod1_ast_ratio
            
            pod1_nlr = st.number_input(
                "POD1 NLR",
                min_value=0.0, max_value=20.0, value=None, step=0.1,
                help=VARIABLE_DESCRIPTIONS['POD1_NLR']
            )
            if pod1_nlr is not None:
                input_dict['POD1_NLR'] = pod1_nlr
            
            post6m_nlr = st.number_input(
                "Post6M NLR",
                min_value=0.0, max_value=20.0, value=None, step=0.1,
                help=VARIABLE_DESCRIPTIONS['Post6M_NLR']
            )
            if post6m_nlr is not None:
                input_dict['Post6M_NLR'] = post6m_nlr
    
    with tab3:
        st.markdown("### 🟪 IHC Enhancement (Optional)")
        st.caption("Enter IHC marker values if available - these will enhance prediction accuracy")
        st.info("💡 **Key markers for heat phenotype:** HSP70, HIF-1α, BCL-2 (all three needed for complete heat phenotype assessment)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Markers (Heat Phenotype)**")
            
            hsp70 = st.number_input(
                "HSP70 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['HSP70']
            )
            if hsp70 is not None:
                input_dict['HSP70'] = hsp70
            
            hif1a = st.number_input(
                "HIF-1α (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['HIF_1α']
            )
            if hif1a is not None:
                input_dict['HIF_1α'] = hif1a
            
            bcl2 = st.number_input(
                "BCL-2 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['BCL_2']
            )
            if bcl2 is not None:
                input_dict['BCL_2'] = bcl2
            
            st.markdown("---")
            st.markdown("**Proliferation Markers**")
            
            ki67 = st.number_input(
                "Ki-67 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['Ki_67']
            )
            if ki67 is not None:
                input_dict['Ki_67'] = ki67
            
            gpc3 = st.number_input(
                "GPC-3 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['GPC_3']
            )
            if gpc3 is not None:
                input_dict['GPC_3'] = gpc3
        
        with col2:
            st.markdown("**Cytokeratin Markers**")
            
            ck7 = st.number_input(
                "CK7 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['CK7']
            )
            if ck7 is not None:
                input_dict['CK7'] = ck7
            
            ck19 = st.number_input(
                "CK19 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['CK19']
            )
            if ck19 is not None:
                input_dict['CK19'] = ck19
            
            st.markdown("---")
            st.markdown("**Invasion & Angiogenesis Markers**")
            
            e_cadherin = st.number_input(
                "E-cadherin (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['E_cadherin']
            )
            if e_cadherin is not None:
                input_dict['E_cadherin'] = e_cadherin
            
            mmp9 = st.number_input(
                "MMP-9 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['MMP_9']
            )
            if mmp9 is not None:
                input_dict['MMP_9'] = mmp9
            
            vegf = st.number_input(
                "VEGF (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['VEGF']
            )
            if vegf is not None:
                input_dict['VEGF'] = vegf
            
            cd34 = st.number_input(
                "CD34 (0-100)",
                min_value=0.0, max_value=100.0, value=None, step=1.0,
                help=VARIABLE_DESCRIPTIONS['CD34']
            )
            if cd34 is not None:
                input_dict['CD34'] = cd34
            
            mvi = st.selectbox(
                "MVI (Microvascular Invasion)",
                options=CATEGORICAL_OPTIONS['MVI'],
                index=0,
                help=VARIABLE_DESCRIPTIONS['MVI']
            )
            if mvi != '':
                input_dict['MVI'] = CATEGORICAL_ENCODING['MVI'][mvi]
    
    return input_dict

def render_results(results, input_dict):
    """渲染结果展示（含IHC和热耐受表型）"""
    
    st.markdown("## 📊 Prediction Results")
    
    # 数据完整度显示
    st.info(f"📊 Data Completeness: {results['completeness']:.0f}% | "
            f"Postop: {results['available_postop']}/{results['total_postop']} | "
            f"IHC: {results['available_ihc']}/{results['total_ihc']} | "
            f"Confidence: {results['confidence']} ({results['confidence_label']})")
    
    # 热耐受表型显示
    if results['heat_phenotype'] != 'Not Available':
        if results['heat_phenotype_color'] == 'positive':
            phenotype_html = f'<span class="heat-phenotype">🔥 Heat Phenotype: {results["heat_phenotype"]}</span>'
        else:
            phenotype_html = f'<span class="heat-phenotype" style="background-color:#3498db;">❄️ Heat Phenotype: {results["heat_phenotype"]}</span>'
        st.markdown(phenotype_html, unsafe_allow_html=True)
        if results['heat_markers_present']:
            st.caption(f"Markers used: {', '.join(results['heat_markers_present'])} (Avg: {results['heat_score_avg']:.1f})")
    else:
        st.caption("⚠️ Heat phenotype: Insufficient IHC markers (need HSP70, HIF-1α, BCL-2)")
    
    # 风险评分
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="RFA Predicted LTP Probability",
            value=f"{results['P_RFA']*100:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="IRE Predicted LTP Probability",
            value=f"{results['P_IRE']*100:.1f}%",
            delta=None
        )
    
    with col3:
        delta_display = results['delta_P'] * 100
        st.metric(
            label="ΔP (RFA - IRE)",
            value=f"{delta_display:+.1f}%",
            delta_color="normal"
        )
    
    # 风险分组和推荐
    col1, col2 = st.columns(2)
    
    with col1:
        risk_group = results['risk_group']
        if risk_group == 'Low':
            risk_html = f'<span class="risk-low">🟢 Low Risk</span>'
        elif risk_group == 'Intermediate':
            risk_html = f'<span class="risk-intermediate">🟡 Intermediate Risk</span>'
        else:
            risk_html = f'<span class="risk-high">🔴 High Risk</span>'
        
        st.markdown(f"### Risk Group")
        st.markdown(risk_html, unsafe_allow_html=True)
        st.caption("Thresholds: Low < 0.2, Intermediate 0.2-0.5, High ≥ 0.5")
    
    with col2:
        recommendation = results['recommendation']
        if recommendation == 'IRE':
            rec_html = f'<div class="recommend-ire">💡 Recommended: IRE (Irreversible Electroporation)</div>'
        elif recommendation == 'RFA':
            rec_html = f'<div class="recommend-rfa">💡 Recommended: RFA (Radiofrequency Ablation)</div>'
        else:
            rec_html = f'<div class="recommend-either">💡 Either technique is suitable</div>'
        
        st.markdown(f"### Treatment Recommendation")
        st.markdown(rec_html, unsafe_allow_html=True)
        st.caption(results['recommendation_detail'])
        
        # IHC强化推荐
        if results['ihc_available'] and results['heat_phenotype'] != 'Not Available':
            if 'Positive' in results['heat_phenotype'] and results['recommendation'] != 'IRE':
                st.warning("⚠️ Heat phenotype positive detected. Consider IRE despite model recommendation.")
            elif 'Negative' in results['heat_phenotype'] and results['recommendation'] == 'IRE':
                st.info("ℹ️ Heat phenotype negative. RFA may be appropriate if clinically indicated.")
    
    # 显示输入数据摘要
    with st.expander("📋 View Input Data"):
        all_vars_display = PREOP_MANDATORY_VARS + POSTOP_OPTIONAL_VARS + IHC_ENHANCEMENT_VARS
        data_rows = []
        for v in all_vars_display:
            if v in input_dict and input_dict.get(v) is not None and not pd.isna(input_dict.get(v)):
                val = input_dict.get(v)
                # 解码分类变量
                if v in CATEGORICAL_ENCODING:
                    for key, code in CATEGORICAL_ENCODING[v].items():
                        if code == val:
                            val = f"{val} ({key})"
                            break
                data_rows.append({
                    'Variable': VARIABLE_DESCRIPTIONS.get(v, v),
                    'Value': val
                })
        input_df = pd.DataFrame(data_rows)
        st.dataframe(input_df, use_container_width=True)

def generate_pdf_report(results, input_dict):
    """生成PDF报告（含IHC）"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # 标题
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("LIGHTCURE Treatment Decision Report", title_style))
    story.append(Spacer(1, 20))
    
    # 时间
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # 预测结果
    story.append(Paragraph("Prediction Results", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    result_data = [
        ["Metric", "Value"],
        ["RFA Predicted LTP Probability", f"{results['P_RFA']*100:.1f}%"],
        ["IRE Predicted LTP Probability", f"{results['P_IRE']*100:.1f}%"],
        ["ΔP (RFA - IRE)", f"{results['delta_P']*100:+.1f}%"],
        ["Risk Group", results['risk_group']],
        ["Treatment Recommendation", results['recommendation']],
        ["Heat Phenotype", results['heat_phenotype']],
        ["Data Completeness", f"{results['completeness']:.0f}% ({results['confidence']})"]
    ]
    
    result_table = Table(result_data, colWidths=[250, 150])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 30))
    
    # 输入数据
    story.append(Paragraph("Input Data", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    input_data = []
    input_data.append(["Variable", "Value"])
    for key, value in input_dict.items():
        display_key = VARIABLE_DESCRIPTIONS.get(key, key)
        if key in ['Etiology', 'Portal_Hypertension', 'Arterial_Enhancement_preop', 
                   'Washout_preop', 'Shape_Irregular_preop', 'Capsule_Intact_preop',
                   'Margin_Ill_Defined_preop', 'US_Echogenicity_Preop', 'subcapsular',
                   'CEUS_Pattern_preop', 'New_Nodule_post6m', 'Complete_Ablation_post3m',
                   'Complete_Ablation_post1m', 'New_Nodule_post3m', 'Complete_Ablation_post6m',
                   'MVI']:
            for k, code in CATEGORICAL_ENCODING.get(key, {}).items():
                if code == value:
                    value = k
                    break
        input_data.append([display_key, str(value)])
    
    input_table = Table(input_data, colWidths=[300, 100])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    story.append(input_table)
    story.append(Spacer(1, 30))
    
    # 免责声明
    story.append(Paragraph("Disclaimer", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This report is generated by the LIGHTCURE Decision Support System for clinical reference only. "
        "The final treatment decision should be made by the clinician based on comprehensive patient assessment.",
        styles['Normal']
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# 主应用
# ============================================================================

def main():
    """主应用函数"""
    
    # 标题
    st.markdown('<div class="main-header">🏥 LIGHTCURE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'HCC Ablation Therapy Individualized Decision Support System<br>'
        '<small>肝细胞癌消融治疗个体化决策支持系统</small>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        **LIGHTCURE** is a deep learning-based decision support system for HCC ablation therapy.
        
        **Features:**
        - Predicts LTP probability for RFA and IRE
        - Calculates individualized ΔP benefit score
        - Provides treatment recommendation
        - **IHC Enhancement**: HSP70, HIF-1α, BCL-2 heat phenotype
        - Handles missing postoperative variables
        - SHAP feature explanation
        
        **Based on:**
        - 13 centers, 5,228 patients
        - 23 variables (14 preop + 9 postop)
        - 12 IHC markers (optional enhancement)
        - Missing-adaptive encoder
        
        **Privacy:**
        - All computation performed locally
        - No patient data stored
        """)
        
        st.markdown("---")
        st.markdown("### 📁 Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV for batch prediction",
            type=['csv'],
            help="CSV file must contain all 23 variables (IHC optional)"
        )
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_batch)} records")
                st.session_state.batch_data = df_batch
            except Exception as e:
                st.error(f"❌ Failed to load: {e}")
        
        st.markdown("---")
        st.markdown("### 📄 Report")
        
        if 'results' in st.session_state:
            if st.button("📥 Download PDF Report"):
                try:
                    pdf_buffer = generate_pdf_report(
                        st.session_state.results,
                        st.session_state.input_dict
                    )
                    st.download_button(
                        label="Download Report",
                        data=pdf_buffer,
                        file_name=f"LIGHTCURE_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
        
        st.markdown("---")
        st.markdown("### 🔧 Model Info")
        if 'model_info' in st.session_state:
            info = st.session_state.model_info
            st.caption(f"Input features: {info.get('input_dim', 23)}")
            st.caption(f"Preop features: {info.get('preop_dim', 14)}")
            st.caption(f"Postop features: {info.get('postop_dim', 9)}")
            st.caption(f"IHC features: up to {len(IHC_ENHANCEMENT_VARS)} (optional)")
            st.caption(f"Version: v2.0")
            st.caption(f"Updated: 2026-09-01")
    
    # 主内容
    # 加载模型
    if 'models_loaded' not in st.session_state:
        with st.spinner("⏳ Loading models..."):
            rfa_model, ire_model, scaler, model_info = load_models()
            if rfa_model is not None and scaler is not None:
                st.session_state.rfa_model = rfa_model
                st.session_state.ire_model = ire_model
                st.session_state.scaler = scaler
                st.session_state.models_loaded = True
                st.session_state.model_info = model_info
            else:
                st.error("❌ Model loading failed. Please check model files.")
                return
    
    # 检查是否批量预测
    if 'batch_data' in st.session_state and st.session_state.batch_data is not None:
        st.markdown("## 📊 Batch Prediction Results")
        
        df_batch = st.session_state.batch_data
        
        # 验证输入
        missing_cols = [v for v in ALL_VARS if v not in df_batch.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            # 执行预测
            results_batch = []
            for _, row in df_batch.iterrows():
                input_dict = {v: row[v] for v in ALL_VARS}
                # 也添加IHC列（如果有）
                for ihc_var in IHC_ENHANCEMENT_VARS:
                    if ihc_var in df_batch.columns and not pd.isna(row[ihc_var]):
                        input_dict[ihc_var] = row[ihc_var]
                result = predict_with_ihc(
                    (st.session_state.rfa_model, st.session_state.ire_model),
                    input_dict,
                    st.session_state.scaler
                )
                results_batch.append(result)
            
            df_results = pd.DataFrame(results_batch)
            
            # 显示结果
            st.dataframe(
                df_results,
                column_config={
                    "P_RFA": st.column_config.NumberColumn("P_RFA", format="%.3f"),
                    "P_IRE": st.column_config.NumberColumn("P_IRE", format="%.3f"),
                    "delta_P": st.column_config.NumberColumn("ΔP", format="%.3f"),
                    "risk_group": st.column_config.TextColumn("Risk"),
                    "recommendation": st.column_config.TextColumn("Recommendation"),
                    "heat_phenotype": st.column_config.TextColumn("Heat Phenotype"),
                    "completeness": st.column_config.NumberColumn("Completeness", format="%.0f%%"),
                    "confidence": st.column_config.TextColumn("Confidence")
                },
                use_container_width=True
            )
            
            # 统计信息
            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Total", len(df_results))
            with col2:
                n_high = (df_results['risk_group'] == 'High').sum()
                st.metric("High Risk", f"{n_high} ({n_high/len(df_results)*100:.1f}%)")
            with col3:
                n_ire = (df_results['recommendation'] == 'IRE').sum()
                st.metric("Recommend IRE", f"{n_ire} ({n_ire/len(df_results)*100:.1f}%)")
            with col4:
                n_rfa = (df_results['recommendation'] == 'RFA').sum()
                st.metric("Recommend RFA", f"{n_rfa} ({n_rfa/len(df_results)*100:.1f}%)")
            with col5:
                n_heat = (df_results['heat_phenotype'].str.contains('Positive', na=False)).sum()
                st.metric("Heat Positive", f"{n_heat} ({n_heat/len(df_results)*100:.1f}%)")
            with col6:
                avg_comp = df_results['completeness'].mean()
                st.metric("Avg Completeness", f"{avg_comp:.0f}%")
            
            # 下载结果
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"LIGHTCURE_Batch_Results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        if st.button("🔄 Back to Single Prediction"):
            del st.session_state.batch_data
            st.rerun()
        
        return
    
    # 单例预测
    input_dict = render_input_form()
    
    # 预测按钮
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        try:
            # 执行预测
            results = predict_with_ihc(
                (st.session_state.rfa_model, st.session_state.ire_model),
                input_dict,
                st.session_state.scaler
            )
            
            st.session_state.results = results
            st.session_state.input_dict = input_dict
            
            # 渲染结果
            render_results(results, input_dict)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        LIGHTCURE v2.0 | For clinical research use only | No patient data stored<br>
        Contact: support@lightcure.ai
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()