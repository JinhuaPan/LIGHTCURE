# -*- coding: utf-8 -*-
"""
LIGHTCURE - 肝细胞癌消融治疗决策支持系统
公开平台部署 - Streamlit应用

依据方案：第十六部分 - 公开平台部署
技术栈：Streamlit + PyTorch + SHAP + Matplotlib

功能模块：
- 输入：21个核心变量
- 输出：RFA/IRE风险评分、推荐治疗
- 解释：SHAP解释图
- 报告：PDF报告生成
- 隐私保护：不存储任何患者数据
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
    page_title="LIGHTCURE - 肝细胞癌消融治疗决策系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 全局设置
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
# 模型定义（与训练时一致）
# ============================================================================

class LIGHTCURE_RFA_Model(nn.Module):
    """RFA模型架构"""
    def __init__(self, input_dim, hidden_dims=[192, 96, 48], dropout=0.45):
        super(LIGHTCURE_RFA_Model, self).__init__()
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
    
    def forward(self, x):
        x = self.bn_input(x)
        x = self.dropout_input(x)
        x = self.hidden(x)
        x = self.output(x)
        return self.sigmoid(x).squeeze(-1)


class LIGHTCURE_IRE_Model(nn.Module):
    """IRE模型架构"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.4):
        super(LIGHTCURE_IRE_Model, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.dropout_input = nn.Dropout(0.2)
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            d_rate = dropout - i * 0.05 if i < len(hidden_dims) - 1 else dropout
            layers.append(nn.Dropout(d_rate))
            prev_dim = h_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.bn_input(x)
        x = self.dropout_input(x)
        x = self.hidden(x)
        x = self.output(x)
        return self.sigmoid(x).squeeze(-1)

# ============================================================================
# 变量定义
# ============================================================================

V_STATIC = [
    "Etiology", "Number_of_lesions", "Maximum_diameter",
    "US_Echogenicity_Preop", "Age", "Arterial_Enhancement_preop",
    "Restricted_Diffusion_preop", "DWI_High_preop", "Portal_Hypertension",
    "CEUS_Pattern_preop", "Capsule_Intact_preop", "subcapsular"
]

V_DELTA = [
    "New_Nodule_post3m", "New_Nodule_post6m",
    "Complete_Ablation_post6m", "Complete_Ablation_post3m",
    "Post6M_NLR", "POD1_NLR", "ALT_Recovery_Ratio_6m", "POD1_AST_Ratio"
]

ALL_VARS = V_STATIC + V_DELTA

# 变量描述（用于UI显示）
VARIABLE_DESCRIPTIONS = {
    "Etiology": "病因 (HBV=1, HCV=2, NAFLD=3, ALD=4, 其他=0)",
    "Number_of_lesions": "肿瘤数目",
    "Maximum_diameter": "肿瘤最大径 (cm)",
    "US_Echogenicity_Preop": "超声回声类型 (低回声=1, 等回声=2, 高回声=3, 混合=4)",
    "Age": "年龄 (岁)",
    "Arterial_Enhancement_preop": "动脉期强化 (是=1, 否=0)",
    "Restricted_Diffusion_preop": "弥散受限 (是=1, 否=0)",
    "DWI_High_preop": "DWI高信号 (是=1, 否=0)",
    "Portal_Hypertension": "门脉高压 (是=1, 否=0)",
    "CEUS_Pattern_preop": "CEUS增强模式 (快进快出=1, 其他=0)",
    "Capsule_Intact_preop": "包膜完整 (是=1, 否=0)",
    "subcapsular": "包膜下肿瘤 (是=1, 否=0)",
    "New_Nodule_post3m": "术后3月新发结节 (是=1, 否=0)",
    "New_Nodule_post6m": "术后6月新发结节 (是=1, 否=0)",
    "Complete_Ablation_post6m": "术后6月完全消融 (是=1, 否=0)",
    "Complete_Ablation_post3m": "术后3月完全消融 (是=1, 否=0)",
    "Post6M_NLR": "术后6月中性粒细胞/淋巴细胞比值",
    "POD1_NLR": "术后1天中性粒细胞/淋巴细胞比值",
    "ALT_Recovery_Ratio_6m": "术后6月ALT恢复比率",
    "POD1_AST_Ratio": "术后1天AST变化率"
}

# 变量范围（用于验证）
VARIABLE_RANGES = {
    "Etiology": (0, 4),
    "Number_of_lesions": (1, 10),
    "Maximum_diameter": (0.1, 5.0),
    "US_Echogenicity_Preop": (1, 4),
    "Age": (18, 85),
    "Arterial_Enhancement_preop": (0, 1),
    "Restricted_Diffusion_preop": (0, 1),
    "DWI_High_preop": (0, 1),
    "Portal_Hypertension": (0, 1),
    "CEUS_Pattern_preop": (0, 1),
    "Capsule_Intact_preop": (0, 1),
    "subcapsular": (0, 1),
    "New_Nodule_post3m": (0, 1),
    "New_Nodule_post6m": (0, 1),
    "Complete_Ablation_post6m": (0, 1),
    "Complete_Ablation_post3m": (0, 1),
    "Post6M_NLR": (0, 20),
    "POD1_NLR": (0, 20),
    "ALT_Recovery_Ratio_6m": (0, 5),
    "POD1_AST_Ratio": (0, 5)
}

# ============================================================================
# 模型加载函数
# ============================================================================

@st.cache_resource
def load_models():
    """加载模型和标准化器"""
    
    # 模型文件路径（按优先级顺序）
    model_paths = {
        'rfa': [
            'models/LIGHTCURE_RFA_Model.pth',
            '../models/LIGHTCURE_RFA_Model.pth',
            'LIGHTCURE_RFA_Model.pth',
        ],
        'ire': [
            'models/LIGHTCURE_IRE_Model_Best.pth',
            '../models/LIGHTCURE_IRE_Model_Best.pth',
            'LIGHTCURE_IRE_Model_Best.pth',
        ],
        'scaler': [
            'models/scaler_rfa.pkl',
            '../models/scaler_rfa.pkl',
            'scaler_rfa.pkl',
        ],
        'model_info': [
            'models/model_info.json',
            '../models/model_info.json',
            'model_info.json',
        ]
    }
    
    # 尝试加载
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
    else:
        model_info = {'input_dim': 20, 'hidden_dims': [192, 96, 48]}
    
    # 加载RFA模型
    rfa_path = find_file(model_paths['rfa'])
    if rfa_path is None:
        st.error("❌ RFA模型文件未找到，请确保模型文件已放置在正确位置")
        return None, None, None, None
    
    rfa_model = LIGHTCURE_RFA_Model(
        input_dim=model_info.get('input_dim', 20),
        hidden_dims=model_info.get('hidden_dims', [192, 96, 48]),
        dropout=model_info.get('dropout', 0.45)
    )
    try:
        rfa_model.load_state_dict(torch.load(rfa_path, map_location='cpu'))
        rfa_model.eval()
        st.success(f"✅ RFA模型加载成功: {rfa_path}")
    except Exception as e:
        st.error(f"❌ RFA模型加载失败: {e}")
        return None, None, None, None
    
    # 加载IRE模型
    ire_path = find_file(model_paths['ire'])
    if ire_path is None:
        st.warning("⚠️ IRE模型文件未找到，将使用RFA模型作为替代")
        ire_model = rfa_model
    else:
        ire_model = LIGHTCURE_IRE_Model(
            input_dim=model_info.get('input_dim', 20),
            hidden_dims=[256, 128, 64],
            dropout=0.4
        )
        try:
            ire_model.load_state_dict(torch.load(ire_path, map_location='cpu'))
            ire_model.eval()
            st.success(f"✅ IRE模型加载成功: {ire_path}")
        except Exception as e:
            st.warning(f"⚠️ IRE模型加载失败: {e}，使用RFA模型作为替代")
            ire_model = rfa_model
    
    # 加载标准化器
    scaler_path = find_file(model_paths['scaler'])
    if scaler_path is None:
        st.error("❌ 标准化器文件未找到")
        return rfa_model, ire_model, None, model_info
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        st.success(f"✅ 标准化器加载成功: {scaler_path}")
    
    return rfa_model, ire_model, scaler, model_info

# ============================================================================
# 预测函数
# ============================================================================

def predict(models, input_data, scaler):
    """
    进行预测
    
    参数:
        models: (rfa_model, ire_model) 元组
        input_data: 输入的DataFrame或numpy数组
        scaler: 标准化器
    
    返回:
        dict: 包含P_RFA, P_IRE, delta_P, risk_group, recommendation
    """
    rfa_model, ire_model = models
    
    # 转换为numpy数组
    if isinstance(input_data, pd.DataFrame):
        X = input_data.values
    else:
        X = np.array(input_data).reshape(1, -1)
    
    # 标准化
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # 预测
    with torch.no_grad():
        P_RFA = rfa_model(X_tensor).numpy().flatten()
        P_IRE = ire_model(X_tensor).numpy().flatten()
    
    delta_P = P_RFA - P_IRE
    
    # 风险分组（根据方案5.5）
    risk_group = []
    for p in P_RFA:
        if p < 0.2:
            risk_group.append('Low')
        elif p < 0.5:
            risk_group.append('Intermediate')
        else:
            risk_group.append('High')
    
    # 治疗推荐（根据方案7.2）
    recommendation = []
    for dp in delta_P:
        if dp > 0:
            recommendation.append('IRE')
        elif dp < 0:
            recommendation.append('RFA')
        else:
            recommendation.append('Either')
    
    return {
        'P_RFA': P_RFA[0],
        'P_IRE': P_IRE[0],
        'delta_P': delta_P[0],
        'risk_group': risk_group[0],
        'recommendation': recommendation[0]
    }

# ============================================================================
# SHAP解释函数
# ============================================================================

@st.cache_resource
def get_shap_explainer(models, scaler, feature_names, background_data):
    """获取SHAP解释器"""
    rfa_model, ire_model = models
    
    # 创建包装函数用于SHAP
    def predict_proba_rfa(X):
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            return rfa_model(X_tensor).numpy().flatten()
    
    # 使用背景数据创建解释器
    background_scaled = scaler.transform(background_data)
    explainer = shap.KernelExplainer(predict_proba_rfa, background_scaled)
    
    return explainer

def get_shap_values(explainer, X, feature_names):
    """获取SHAP值"""
    X_scaled = scaler.transform(X)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values

# ============================================================================
# 渲染函数
# ============================================================================

def render_input_form():
    """渲染输入表单"""
    
    st.markdown("## 📋 患者信息输入")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 术前特征 (T0)")
        
        age = st.number_input(
            "年龄 (岁)",
            min_value=18, max_value=85, value=55,
            help=VARIABLE_DESCRIPTIONS['Age']
        )
        
        etiology = st.selectbox(
            "病因",
            options=['HBV', 'HCV', 'NAFLD', 'ALD', '其他'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Etiology']
        )
        etiology_map = {'HBV': 1, 'HCV': 2, 'NAFLD': 3, 'ALD': 4, '其他': 0}
        etiology_code = etiology_map[etiology]
        
        num_lesions = st.number_input(
            "肿瘤数目",
            min_value=1, max_value=10, value=1,
            help=VARIABLE_DESCRIPTIONS['Number_of_lesions']
        )
        
        max_diameter = st.number_input(
            "肿瘤最大径 (cm)",
            min_value=0.1, max_value=5.0, value=2.0, step=0.1,
            help=VARIABLE_DESCRIPTIONS['Maximum_diameter']
        )
        
        us_echogenicity = st.selectbox(
            "超声回声类型",
            options=['低回声', '等回声', '高回声', '混合'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['US_Echogenicity_Preop']
        )
        us_map = {'低回声': 1, '等回声': 2, '高回声': 3, '混合': 4}
        us_code = us_map[us_echogenicity]
        
        arterial_enhancement = st.selectbox(
            "动脉期强化",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Arterial_Enhancement_preop']
        )
        arterial_code = 1 if arterial_enhancement == '是' else 0
        
        restricted_diffusion = st.selectbox(
            "弥散受限",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Restricted_Diffusion_preop']
        )
        restricted_code = 1 if restricted_diffusion == '是' else 0
        
        dwi_high = st.selectbox(
            "DWI高信号",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['DWI_High_preop']
        )
        dwi_code = 1 if dwi_high == '是' else 0
        
        portal_hypertension = st.selectbox(
            "门脉高压",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Portal_Hypertension']
        )
        portal_code = 1 if portal_hypertension == '是' else 0
        
        ceus_pattern = st.selectbox(
            "CEUS增强模式",
            options=['快进快出', '其他'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['CEUS_Pattern_preop']
        )
        ceus_code = 1 if ceus_pattern == '快进快出' else 0
        
        capsule_intact = st.selectbox(
            "包膜完整",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Capsule_Intact_preop']
        )
        capsule_code = 1 if capsule_intact == '是' else 0
        
        subcapsular = st.selectbox(
            "包膜下肿瘤",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['subcapsular']
        )
        subcapsular_code = 1 if subcapsular == '是' else 0
    
    with col2:
        st.markdown("### 术后动态特征")
        
        new_nodule_3m = st.selectbox(
            "术后3月新发结节",
            options=['是', '否'],
            index=1,
            help=VARIABLE_DESCRIPTIONS['New_Nodule_post3m']
        )
        new_nodule_3m_code = 1 if new_nodule_3m == '是' else 0
        
        new_nodule_6m = st.selectbox(
            "术后6月新发结节",
            options=['是', '否'],
            index=1,
            help=VARIABLE_DESCRIPTIONS['New_Nodule_post6m']
        )
        new_nodule_6m_code = 1 if new_nodule_6m == '是' else 0
        
        complete_ablation_3m = st.selectbox(
            "术后3月完全消融",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Complete_Ablation_post3m']
        )
        complete_ablation_3m_code = 1 if complete_ablation_3m == '是' else 0
        
        complete_ablation_6m = st.selectbox(
            "术后6月完全消融",
            options=['是', '否'],
            index=0,
            help=VARIABLE_DESCRIPTIONS['Complete_Ablation_post6m']
        )
        complete_ablation_6m_code = 1 if complete_ablation_6m == '是' else 0
        
        pod1_nlr = st.number_input(
            "术后1天NLR",
            min_value=0.0, max_value=20.0, value=2.5, step=0.1,
            help=VARIABLE_DESCRIPTIONS['POD1_NLR']
        )
        
        post6m_nlr = st.number_input(
            "术后6月NLR",
            min_value=0.0, max_value=20.0, value=2.0, step=0.1,
            help=VARIABLE_DESCRIPTIONS['Post6M_NLR']
        )
        
        alt_recovery_ratio = st.number_input(
            "术后6月ALT恢复比率",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help=VARIABLE_DESCRIPTIONS['ALT_Recovery_Ratio_6m']
        )
        
        pod1_ast_ratio = st.number_input(
            "术后1天AST变化率",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help=VARIABLE_DESCRIPTIONS['POD1_AST_Ratio']
        )
    
    # 构建输入数据
    input_dict = {
        'Etiology': etiology_code,
        'Number_of_lesions': num_lesions,
        'Maximum_diameter': max_diameter,
        'US_Echogenicity_Preop': us_code,
        'Age': age,
        'Arterial_Enhancement_preop': arterial_code,
        'Restricted_Diffusion_preop': restricted_code,
        'DWI_High_preop': dwi_code,
        'Portal_Hypertension': portal_code,
        'CEUS_Pattern_preop': ceus_code,
        'Capsule_Intact_preop': capsule_code,
        'subcapsular': subcapsular_code,
        'New_Nodule_post3m': new_nodule_3m_code,
        'New_Nodule_post6m': new_nodule_6m_code,
        'Complete_Ablation_post6m': complete_ablation_6m_code,
        'Complete_Ablation_post3m': complete_ablation_3m_code,
        'Post6M_NLR': post6m_nlr,
        'POD1_NLR': pod1_nlr,
        'ALT_Recovery_Ratio_6m': alt_recovery_ratio,
        'POD1_AST_Ratio': pod1_ast_ratio
    }
    
    return input_dict

def render_results(results, input_dict):
    """渲染结果展示"""
    
    st.markdown("## 📊 预测结果")
    
    # 风险评分
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="RFA预测LTP概率",
            value=f"{results['P_RFA']*100:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="IRE预测LTP概率",
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
            risk_html = f'<span class="risk-low">🟢 低危 (Low)</span>'
        elif risk_group == 'Intermediate':
            risk_html = f'<span class="risk-intermediate">🟡 中危 (Intermediate)</span>'
        else:
            risk_html = f'<span class="risk-high">🔴 高危 (High)</span>'
        
        st.markdown(f"### 风险分组")
        st.markdown(risk_html, unsafe_allow_html=True)
        st.caption("阈值: Low < 0.2, Intermediate 0.2-0.5, High ≥ 0.5")
    
    with col2:
        recommendation = results['recommendation']
        if recommendation == 'IRE':
            rec_html = f'<div class="recommend-ire">💡 推荐: 不可逆电穿孔 (IRE)</div>'
        elif recommendation == 'RFA':
            rec_html = f'<div class="recommend-rfa">💡 推荐: 射频消融 (RFA)</div>'
        else:
            rec_html = f'<div class="recommend-either">💡 两种技术均可</div>'
        
        st.markdown(f"### 治疗推荐")
        st.markdown(rec_html, unsafe_allow_html=True)
        
        if results['delta_P'] > 0:
            st.caption(f"ΔP > 0: IRE预期可降低 {results['delta_P']*100:.1f}% 的LTP风险")
        elif results['delta_P'] < 0:
            st.caption(f"ΔP < 0: RFA预期可降低 {-results['delta_P']*100:.1f}% 的LTP风险")
        else:
            st.caption("ΔP ≈ 0: 两种技术预期效果相似")
    
    # SHAP解释
    st.markdown("## 🔍 特征贡献解释")
    st.caption("SHAP值解释: 红色表示该特征增加风险，蓝色表示降低风险")
    
    if 'shap_values' in st.session_state and st.session_state.shap_values is not None:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                st.session_state.shap_values,
                np.array([list(input_dict.values())]),
                feature_names=list(input_dict.keys()),
                show=False,
                max_display=15
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP图生成失败: {e}")
    else:
        st.info("点击「生成SHAP解释」按钮查看特征贡献图")
        
        if st.button("📊 生成SHAP解释"):
            try:
                # 创建背景数据
                background_data = np.random.randn(100, len(ALL_VARS))
                explainer = shap.KernelExplainer(
                    lambda x: predict_proba_rfa(x, rfa_model, scaler),
                    background_data
                )
                X_input = np.array([list(input_dict.values())])
                shap_values = explainer.shap_values(X_input)
                
                st.session_state.shap_values = shap_values
                st.rerun()
            except Exception as e:
                st.error(f"SHAP解释生成失败: {e}")

def predict_proba_rfa(X, rfa_model, scaler):
    """用于SHAP的预测函数"""
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        return rfa_model(X_tensor).numpy().flatten()

def generate_pdf_report(results, input_dict):
    """生成PDF报告"""
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
    story.append(Paragraph("LIGHTCURE 治疗决策报告", title_style))
    story.append(Spacer(1, 20))
    
    # 时间
    story.append(Paragraph(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # 预测结果
    story.append(Paragraph("预测结果", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    result_data = [
        ["指标", "数值"],
        ["RFA预测LTP概率", f"{results['P_RFA']*100:.1f}%"],
        ["IRE预测LTP概率", f"{results['P_IRE']*100:.1f}%"],
        ["ΔP (RFA - IRE)", f"{results['delta_P']*100:+.1f}%"],
        ["风险分组", results['risk_group']],
        ["治疗推荐", results['recommendation']]
    ]
    
    result_table = Table(result_data, colWidths=[200, 200])
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
    story.append(Paragraph("患者输入数据", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    input_data = []
    input_data.append(["变量", "值"])
    for key, value in input_dict.items():
        input_data.append([VARIABLE_DESCRIPTIONS.get(key, key), str(value)])
    
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
    story.append(Paragraph("免责声明", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "本报告由LIGHTCURE决策支持系统生成，仅供临床参考。最终治疗决策应由临床医生根据患者具体情况综合判断。",
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
        '肝细胞癌消融治疗个体化决策支持系统<br>'
        '<small>HCC Ablation Therapy Decision Support System</small>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## ℹ️ 关于系统")
        st.markdown("""
        **LIGHTCURE** 是一个基于深度学习的肝细胞癌治疗决策支持系统。
        
        **功能:**
        - 预测RFA和IRE的LTP概率
        - 计算个体化ΔP获益评分
        - 提供治疗推荐
        - SHAP特征解释
        
        **依据方案:**
        - 13个中心, 5,228例患者
        - 21个核心变量
        - 多模态时序预测
        
        **隐私保护:**
        - 所有计算在本地完成
        - 不存储任何患者数据
        """)
        
        st.markdown("---")
        st.markdown("### 📁 数据导入")
        
        uploaded_file = st.file_uploader(
            "上传CSV文件进行批量预测",
            type=['csv'],
            help="CSV文件需包含所有21个变量"
        )
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ 成功加载 {len(df_batch)} 条记录")
                st.session_state.batch_data = df_batch
            except Exception as e:
                st.error(f"❌ 文件加载失败: {e}")
        
        st.markdown("---")
        st.markdown("### 📄 报告下载")
        
        if 'results' in st.session_state:
            if st.button("📥 下载PDF报告"):
                try:
                    pdf_buffer = generate_pdf_report(
                        st.session_state.results,
                        st.session_state.input_dict
                    )
                    st.download_button(
                        label="点击下载报告",
                        data=pdf_buffer,
                        file_name=f"LIGHTCURE_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"报告生成失败: {e}")
        
        st.markdown("---")
        st.markdown("### 🔧 模型信息")
        if 'model_info' in st.session_state:
            info = st.session_state.model_info
            st.caption(f"输入特征数: {info.get('input_dim', 'N/A')}")
            st.caption(f"模型版本: v1.0")
            st.caption(f"最后更新: 2026-12-01")
    
    # 主内容
    # 加载模型
    if 'models_loaded' not in st.session_state:
        with st.spinner("⏳ 加载模型中..."):
            rfa_model, ire_model, scaler, model_info = load_models()
            if rfa_model is not None and scaler is not None:
                st.session_state.rfa_model = rfa_model
                st.session_state.ire_model = ire_model
                st.session_state.scaler = scaler
                st.session_state.models_loaded = True
                st.session_state.model_info = model_info
            else:
                st.error("❌ 模型加载失败，请检查模型文件")
                return
    
    # 检查是否批量预测
    if 'batch_data' in st.session_state and st.session_state.batch_data is not None:
        st.markdown("## 📊 批量预测结果")
        
        df_batch = st.session_state.batch_data
        
        # 验证输入
        missing_cols = [v for v in ALL_VARS if v not in df_batch.columns]
        if missing_cols:
            st.error(f"❌ 缺少列: {missing_cols}")
        else:
            # 执行预测
            results_batch = []
            for _, row in df_batch.iterrows():
                input_data = row[ALL_VARS].values.reshape(1, -1)
                result = predict(
                    (st.session_state.rfa_model, st.session_state.ire_model),
                    input_data,
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
                    "risk_group": st.column_config.TextColumn("风险分组"),
                    "recommendation": st.column_config.TextColumn("推荐治疗")
                },
                use_container_width=True
            )
            
            # 统计信息
            st.markdown("### 📈 统计摘要")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总样本", len(df_results))
            with col2:
                n_high = (df_results['risk_group'] == 'High').sum()
                st.metric("高危患者", f"{n_high} ({n_high/len(df_results)*100:.1f}%)")
            with col3:
                n_ire = (df_results['recommendation'] == 'IRE').sum()
                st.metric("推荐IRE", f"{n_ire} ({n_ire/len(df_results)*100:.1f}%)")
            with col4:
                n_rfa = (df_results['recommendation'] == 'RFA').sum()
                st.metric("推荐RFA", f"{n_rfa} ({n_rfa/len(df_results)*100:.1f}%)")
            
            # 下载结果
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 下载预测结果CSV",
                data=csv,
                file_name=f"LIGHTCURE_Batch_Results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        if st.button("🔄 返回单例预测"):
            del st.session_state.batch_data
            st.rerun()
        
        return
    
    # 单例预测
    input_dict = render_input_form()
    
    # 预测按钮
    if st.button("🚀 开始预测", type="primary", use_container_width=True):
        try:
            # 构建输入数据
            input_values = [input_dict[v] for v in ALL_VARS]
            input_array = np.array(input_values).reshape(1, -1)
            
            # 执行预测
            results = predict(
                (st.session_state.rfa_model, st.session_state.ire_model),
                input_array,
                st.session_state.scaler
            )
            
            st.session_state.results = results
            st.session_state.input_dict = input_dict
            
            # 渲染结果
            render_results(results, input_dict)
            
            # 显示输入数据摘要
            with st.expander("📋 查看输入数据"):
                input_df = pd.DataFrame({
                    '变量': [VARIABLE_DESCRIPTIONS.get(v, v) for v in ALL_VARS],
                    '值': [input_dict[v] for v in ALL_VARS]
                })
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ 预测失败: {e}")
            st.exception(e)
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        LIGHTCURE v1.0 | 仅供临床研究参考 | 不存储任何患者数据<br>
        如有问题请联系: support@lightcure.ai
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()