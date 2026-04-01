import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Monte Carlo: Proyek Gedung FITE (Advanced)",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling Custom - Level Enterprise Dashboard
st.markdown("""
<style>
    /* Global Typography & Spacing */
    .main-header { font-size: 2.5rem; color: #0f172a; text-align: center; margin-bottom: 0.5rem; font-weight: 800; letter-spacing: -0.5px;}
    .sub-header { font-size: 1.5rem; color: #0f172a; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #f1f5f9; font-weight: 600;}
    
    /* Info & Alert Boxes */
    .info-box { background: linear-gradient(to right, #f0f9ff, #e0f2fe); padding: 1.2rem 1.5rem; border-radius: 12px; border-left: 6px solid #0284c7; margin-bottom: 2rem; color: #0c4a6e; font-size: 1.05rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);}
    
    /* Metric Cards Modernization */
    .metric-container { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
    .metric-card { 
        flex: 1; min-width: 200px;
        background: #ffffff; 
        padding: 1.5rem; 
        border-radius: 16px; 
        text-align: center; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
    .metric-card h3 { font-size: 2.2rem; margin: 0; color: #0ea5e9; font-weight: 800; line-height: 1.2; }
    .metric-card p { margin: 0.5rem 0 0 0; font-size: 0.95rem; color: #64748b; font-weight: 500; }
    .metric-card.danger h3 { color: #ef4444; }
    .metric-card.warning h3 { color: #f59e0b; }
    .metric-card.success h3 { color: #10b981; }

    /* Executive Decision Box */
    .decision-box { padding: 1.5rem; border-radius: 12px; margin-top: 1rem; color: white; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .decision-box.approved { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .decision-box.rejected { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .decision-box h2 { margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 800; color: white; }
    .decision-box p { margin: 0; font-size: 1.1rem; opacity: 0.95; }
    
    /* Sidebar Styling Override */
    [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. KELAS PEMODELAN SISTEM
# ==========================================================
class ProjectStage:
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []

    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        base_duration = np.random.triangular(
            self.optimistic, self.most_likely, self.pessimistic, n_simulations
        )
        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                probability = risk_params['probability']
                impact = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(risk_occurs, base_duration * (1 + impact), base_duration)
            elif risk_params['type'] == 'continuous':
                mean = risk_params['mean']
                std = risk_params['std']
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)
        return base_duration * risk_multiplier

class MonteCarloProjectSimulation:
    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self.initialize_stages()

    def initialize_stages(self):
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = ProjectStage(
                name=stage_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )

    def run_simulation(self, resource_multiplier={}):
        results = pd.DataFrame(index=range(self.num_simulations))
        
        for stage_name, stage in self.stages.items():
            multiplier = resource_multiplier.get(stage_name, 1.0)
            results[stage_name] = stage.sample_duration(self.num_simulations, multiplier)

        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))

        for stage_name in self.stages.keys():
            deps = self.stages[stage_name].dependencies
            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)
            end_times[stage_name] = start_times[stage_name] + results[stage_name]

        results['Total_Duration'] = end_times.max(axis=1)
        
        for stage_name in self.stages.keys():
            results[f'{stage_name}_Finish'] = end_times[stage_name]
            results[f'{stage_name}_Start'] = start_times[stage_name]
            
        self.simulation_results = results
        return results

    def calculate_critical_path_probability(self):
        critical_path_probs = {}
        total_duration = self.simulation_results['Total_Duration']
        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{stage_name}_Finish']
            is_critical = (stage_finish + 0.1) >= total_duration
            prob_critical = np.mean(is_critical)
            critical_path_probs[stage_name] = {
                'probability': prob_critical,
                'avg_duration': self.simulation_results[stage_name].mean()
            }
        return pd.DataFrame(critical_path_probs).T

    def analyze_risk_contribution(self):
        total_var = self.simulation_results['Total_Duration'].var()
        contributions = {}
        for stage_name in self.stages.keys():
            stage_var = self.simulation_results[stage_name].var()
            stage_covar = self.simulation_results[stage_name].cov(self.simulation_results['Total_Duration'])
            contribution = (stage_covar / total_var) * 100
            contributions[stage_name] = {
                'contribution_percent': contribution,
                'std_dev': np.sqrt(stage_var)
            }
        return pd.DataFrame(contributions).T

# ==========================================================
# 3. FUNGSI VISUALISASI PLOTLY (ENHANCED UX)
# ==========================================================
def apply_common_layout(fig):
    """Menerapkan styling korporat yang bersih pada semua grafik Plotly"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans-serif", color="#334155"),
        margin=dict(t=40, b=30, l=30, r=30),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="sans-serif")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
    return fig

def create_distribution_plot(results):
    total_duration = results['Total_Duration']
    mean_duration = total_duration.mean()
    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total_duration, nbinsx=60, name='Distribusi', 
        marker_color='#38bdf8', opacity=0.8, histnorm='probability density',
        hovertemplate='Durasi: %{x:.0f} Hari<br>Densitas: %{y:.4f}<extra></extra>'
    ))
    
    # Garis Mean
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text=f"Mean: {mean_duration:.1f} Hari", annotation_position="top right", annotation_font_color="#ef4444")
    
    # Area CI
    fig.add_vrect(x0=ci_80[0], x1=ci_80[1], fillcolor="#fde047", opacity=0.15, line_width=0, annotation_text="80% CI", annotation_position="top left")
    fig.add_vrect(x0=ci_95[0], x1=ci_95[1], fillcolor="#f97316", opacity=0.1, line_width=0)
    
    fig.update_layout(title='<b>Distribusi Waktu Total Penyelesaian</b>', xaxis_title='Durasi (Hari)', yaxis_title='Densitas', height=400)
    return apply_common_layout(fig)

def create_completion_probability_plot(results):
    total_duration = results['Total_Duration']
    deadlines = np.arange(int(total_duration.min()) - 10, int(total_duration.max()) + 20, 10)
    completion_probs = [np.mean(total_duration <= dl) for dl in deadlines]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deadlines, y=completion_probs, mode='lines', name='Probabilitas', 
        line=dict(color='#0ea5e9', width=3), fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)',
        hovertemplate='Deadline: %{x} Hari<br>Probabilitas: %{y:.1%}<extra></extra>'
    ))
    
    targets = [480, 600, 720]
    colors = ['#f43f5e', '#f59e0b', '#10b981'] # Merah, Kuning, Hijau
    for idx, target in enumerate(targets):
        prob = np.mean(total_duration <= target)
        fig.add_trace(go.Scatter(
            x=[target], y=[prob], mode='markers+text', 
            marker=dict(size=12, color=colors[idx], line=dict(width=2, color='white')), 
            text=[f'<b>{target} Hari ({prob:.1%})</b>'], textposition="top left", showlegend=False
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="#cbd5e1", annotation_text="50% Threshold", annotation_position="bottom right")
    fig.add_hline(y=0.8, line_dash="dash", line_color="#86efac", annotation_text="80% Aman", annotation_position="bottom right")
    
    fig.update_layout(title='<b>Kurva Probabilitas Sukses Deadline</b>', xaxis_title='Deadline (Hari)', yaxis_title='Probabilitas Selesai (%)', yaxis_range=[0, 1.1], height=400)
    return apply_common_layout(fig)

def create_critical_path_plot(critical_analysis):
    df = critical_analysis.sort_values('probability', ascending=True)
    fig = go.Figure()
    
    # Gradasi warna berdasarkan seberapa kritis
    colors = ['#ef4444' if p > 0.6 else '#fb923c' if p > 0.3 else '#94a3b8' for p in df['probability']]
    
    fig.add_trace(go.Bar(
        y=[s[2:].replace('_', ' ') for s in df.index], x=df['probability'], orientation='h', 
        marker_color=colors, marker_line_color='rgba(0,0,0,0)',
        text=[f'<b>{p:.1%}</b>' for p in df['probability']], textposition='auto', textfont=dict(color='white')
    ))
    
    fig.add_vline(x=0.5, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(title='<b>Analisis Kerentanan Jalur Kritis (Critical Path)</b>', xaxis_title='Probabilitas Menjadi Bottleneck', xaxis_range=[0, 1.0], height=400)
    return apply_common_layout(fig)

def create_gantt_chart(results, stages):
    task_data = []
    for stage in stages.keys():
        start_avg = results[f'{stage}_Start'].mean()
        duration_avg = results[stage].mean()
        finish_avg = start_avg + duration_avg
        
        start_date = pd.Timestamp.today() + pd.Timedelta(days=start_avg)
        end_date = pd.Timestamp.today() + pd.Timedelta(days=finish_avg)
        
        task_data.append({
            'Task': stage[2:].replace('_', ' '),
            'Start': start_date,
            'Finish': end_date,
            'Duration': round(duration_avg, 1)
        })
        
    df = pd.DataFrame(task_data)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", text="Duration", title="<b>Gantt Chart (Rata-rata Simulasi Baseline)</b>", color="Duration", color_continuous_scale=px.colors.sequential.Blues)
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=400, coloraxis_showscale=False)
    return apply_common_layout(fig)

# ==========================================================
# 4. FUNGSI UTAMA STREAMLIT
# ==========================================================
def main():
    st.markdown('<div class="main-header">🏗️ Intelijen Proyek Gedung FITE</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#64748b; font-size:1.1rem; margin-bottom:2rem;">Sistem Pendukung Keputusan Berbasis Simulasi Monte Carlo (Studi Kasus 2.1)</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <b>Konteks Manajerial:</b> Dasbor ini mensimulasikan ketidakpastian proyek secara stokastik. Gunakan model ini untuk memetakan probabilitas jadwal, mengidentifikasi <i>bottleneck</i> kritis, mengukur eksposur denda finansial, dan menguji Cost-Benefit dari intervensi <i>resource</i> tambahan.
    </div>
    """, unsafe_allow_html=True)

    # ================= SIDEBAR KONFIGURASI DINAMIS =================
    with st.sidebar:
        st.markdown('<h2>⚙️ Parameter Simulasi</h2>', unsafe_allow_html=True)
        num_simulations = st.slider('Jumlah Iterasi Monte Carlo:', min_value=1000, max_value=20000, value=10000, step=1000)
        st.markdown("---")
        
        default_config = {
            "1_Perencanaan_Desain": {
                "base_params": {"optimistic": 30, "most_likely": 45, "pessimistic": 60},
                "risk_factors": {"perubahan_desain_lab": {"type": "discrete", "probability": 0.3, "impact": 0.2}}
            },
            "2_Pengadaan_Material": {
                "base_params": {"optimistic": 45, "most_likely": 60, "pessimistic": 90},
                "risk_factors": {"telat_material_khusus": {"type": "discrete", "probability": 0.25, "impact": 0.3}},
                "dependencies": ["1_Perencanaan_Desain"]
            },
            "3_Pondasi_Struktur_5L": {
                "base_params": {"optimistic": 120, "most_likely": 150, "pessimistic": 200},
                "risk_factors": {"cuaca_buruk": {"type": "discrete", "probability": 0.4, "impact": 0.25}},
                "dependencies": ["2_Pengadaan_Material"]
            },
            "4_Instalasi_MEP_Lab": {
                "base_params": {"optimistic": 90, "most_likely": 120, "pessimistic": 160},
                "risk_factors": {"kompleksitas_teknis": {"type": "discrete", "probability": 0.3, "impact": 0.15}},
                "dependencies": ["3_Pondasi_Struktur_5L"]
            },
            "5_Finishing_Interior": {
                "base_params": {"optimistic": 60, "most_likely": 80, "pessimistic": 120},
                "risk_factors": {"telat_interior": {"type": "discrete", "probability": 0.15, "impact": 0.2}},
                "dependencies": ["4_Instalasi_MEP_Lab"]
            },
            "6_Testing_Handover": {
                "base_params": {"optimistic": 20, "most_likely": 30, "pessimistic": 45},
                "dependencies": ["5_Finishing_Interior"]
            }
        }

        st.markdown('<h3>⏱️ Kalibrasi Durasi & Risiko</h3>', unsafe_allow_html=True)
        for stage_name, config in default_config.items():
            with st.expander(f"🔹 {stage_name[2:].replace('_', ' ')}", expanded=False):
                st.caption("Estimasi Durasi (Hari)")
                col1, col2, col3 = st.columns(3)
                with col1: opt = st.number_input("Opt", value=config['base_params']['optimistic'], key=f"opt_{stage_name}")
                with col2: ml = st.number_input("ML", value=config['base_params']['most_likely'], key=f"ml_{stage_name}")
                with col3: pes = st.number_input("Pes", value=config['base_params']['pessimistic'], key=f"pes_{stage_name}")
                default_config[stage_name]['base_params'] = {'optimistic': opt, 'most_likely': ml, 'pessimistic': pes}
                
                if 'risk_factors' in config:
                    st.caption("Faktor Risiko")
                    for r_name, r_params in config['risk_factors'].items():
                        if r_params['type'] == 'discrete':
                            new_prob = st.slider(f"Probabilitas {r_name.replace('_', ' ')}", 0.0, 1.0, float(r_params['probability']), key=f"prob_{stage_name}_{r_name}")
                            new_impact = st.slider(f"Impact (% Delay)", 0.0, 1.0, float(r_params['impact']), key=f"imp_{stage_name}_{r_name}")
                            default_config[stage_name]['risk_factors'][r_name]['probability'] = new_prob
                            default_config[stage_name]['risk_factors'][r_name]['impact'] = new_impact

        st.markdown("---")
        st.markdown('<h3>💰 Asumsi Komersial</h3>', unsafe_allow_html=True)
        daily_penalty = st.number_input("Denda Keterlambatan/Hari (Rp)", min_value=0, value=5000000, step=1000000)

        run_sim = st.button("🚀 JALANKAN ANALISIS", type="primary", use_container_width=True)

    if 'base_results' not in st.session_state: st.session_state.base_results = None
    if 'simulator' not in st.session_state: st.session_state.simulator = None

    if run_sim:
        with st.spinner("Memproses ribuan iterasi probabilitas..."):
            sim = MonteCarloProjectSimulation(stages_config=default_config, num_simulations=num_simulations)
            res = sim.run_simulation()
            st.session_state.base_results = res
            st.session_state.simulator = sim
        st.toast('Selesai! Dasbor telah diperbarui.', icon='✅')

    # ================= KONTEN UTAMA =================
    if st.session_state.base_results is not None:
        res = st.session_state.base_results
        sim = st.session_state.simulator
        tot_dur = res['Total_Duration']
        
        # Metrik Makro Bergaya Enterprise
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <h3>{tot_dur.mean():.0f} <span style="font-size:1.2rem; color:#64748b;">Hari</span></h3>
                <p>Ekspektasi Durasi Rata-rata</p>
            </div>
            <div class="metric-card success">
                <h3>{np.percentile(tot_dur, 80):.0f} <span style="font-size:1.2rem; color:#64748b;">Hari</span></h3>
                <p>Jadwal Rekomendasi (Keamanan 80%)</p>
            </div>
            <div class="metric-card warning">
                <h3>{tot_dur.max():.0f} <span style="font-size:1.2rem; color:#64748b;">Hari</span></h3>
                <p>Skenario Terburuk (Maximum)</p>
            </div>
            <div class="metric-card danger">
                <h3>{np.mean(tot_dur > 600) * 100:.1f}%</h3>
                <p>Risiko Melampaui Target 20 Bulan</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sub-header">📈 Intelijen Eksekusi Proyek</div>', unsafe_allow_html=True)
        
        # Styling tab default Streamlit
        css_tabs = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1rem; font-weight: 600; padding: 0.5rem 1rem; }
            .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        </style>
        '''
        st.markdown(css_tabs, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📌 Baseline & Sekuensial", "⚠️ Analisis Kerentanan", "🎯 Eksposur Finansial", "⚙️ Keputusan Investasi (CBA)"])
        
        with tab1:
            st.plotly_chart(create_gantt_chart(res, sim.stages), use_container_width=True)
            st.plotly_chart(create_distribution_plot(res), use_container_width=True)

        with tab2:
            st.write("Fokuskan pengawasan pada tahapan dengan probabilitas tertinggi menjadi hambatan utama (*bottleneck*).")
            c1, c2 = st.columns(2)
            with c1:
                crit_path = sim.calculate_critical_path_probability()
                st.plotly_chart(create_critical_path_plot(crit_path), use_container_width=True)
            with c2:
                risk_contrib = sim.analyze_risk_contribution()
                df_contrib = risk_contrib.sort_values('contribution_percent', ascending=False)
                fig_contrib = go.Figure(go.Bar(
                    x=[s[2:].replace('_', ' ') for s in df_contrib.index], y=df_contrib['contribution_percent'], 
                    marker_color='#8b5cf6', text=[f'{c:.1f}%' for c in df_contrib['contribution_percent']], textposition='auto'
                ))
                fig_contrib.update_layout(title='<b>Kontribusi Risiko terhadap Variabilitas Total</b>', yaxis_title='Kontribusi (%)', height=400)
                st.plotly_chart(apply_common_layout(fig_contrib), use_container_width=True)
                
        with tab3:
            st.plotly_chart(create_completion_probability_plot(res), use_container_width=True)
            
            st.markdown("#### 💸 Kalkulasi Eksposur Risiko Denda")
            st.caption(f"Basis Asumsi: Rp {daily_penalty:,.0f} per Hari Keterlambatan")
            
            c1, c2, c3 = st.columns(3)
            targets = [(480, "Skenario Agresif (16 Bulan)"), (600, "Skenario Moderat (20 Bulan)"), (720, "Skenario Konservatif (24 Bulan)")]
            
            for idx, (target_days, label) in enumerate(targets):
                prob_success = np.mean(tot_dur <= target_days)
                prob_fail = 1 - prob_success
                
                failed_simulations = tot_dur[tot_dur > target_days]
                avg_delay = failed_simulations.mean() - target_days if len(failed_simulations) > 0 else 0
                exp_loss = avg_delay * daily_penalty * prob_fail
                
                # Visual styling untuk metrik
                box_color = "#10b981" if prob_success > 0.8 else "#f59e0b" if prob_success > 0.4 else "#ef4444"
                
                with [c1, c2, c3][idx]:
                    st.markdown(f"""
                    <div style="border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; background:white; border-top: 5px solid {box_color};">
                        <h4 style="margin:0 0 10px 0; color:#334155;">{label}</h4>
                        <div style="font-size:2rem; font-weight:bold; color:{box_color};">{prob_success:.1%}</div>
                        <p style="margin:0; color:#64748b; font-size:0.9rem;">Peluang Kesuksesan</p>
                        <hr style="margin:10px 0;">
                        <div style="font-size:1.2rem; font-weight:bold; color:#ef4444;">Rp {exp_loss/1e6:,.1f} Juta</div>
                        <p style="margin:0; color:#64748b; font-size:0.85rem;">Ekspektasi Denda (Expected Loss)</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab4:
            st.markdown("### ⚙️ Kalkulator Investasi Akselerasi (Cost-Benefit Analysis)")
            st.write("Evaluasi apakah biaya *crashing* proyek (menambah resource) sepadan dengan penghindaran denda keterlambatan.")
            
            resource_db = {
                "Tim Pekerja Tambahan (Shift Malam)": {"cost_per_day": 2000000, "efficiency_gain": 0.15},
                "Sewa Alat Berat (Excavator/Crane)": {"cost_per_day": 3500000, "efficiency_gain": 0.25},
                "Insinyur Senior Resolusi Konflik": {"cost_per_day": 1500000, "efficiency_gain": 0.10}
            }

            most_critical = crit_path.sort_values('probability', ascending=False).index[0]
            
            st.markdown("""<div style="background:#f8fafc; padding:1.5rem; border-radius:10px; border:1px solid #e2e8f0; margin-bottom:1.5rem;">""", unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns([1.5, 2, 1, 1])
            with cc1:
                target_stage = st.selectbox("Target Intervensi", list(sim.stages.keys()), index=list(sim.stages.keys()).index(most_critical))
            with cc2:
                resource_type = st.selectbox("Jenis Resource", list(resource_db.keys()))
            with cc3:
                qty = st.number_input("Unit/Tim", min_value=1, max_value=5, value=1)
            with cc4:
                eval_target = st.number_input("Target Aman (Hari)", min_value=400, max_value=800, value=600, step=10)
            
            run_opt = st.button("HITUNG ROI INVESTASI", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
                
            if run_opt:
                with st.spinner("Mengkalkulasi model finansial..."):
                    res_params = resource_db[resource_type]
                    total_efficiency = min(res_params["efficiency_gain"] * qty, 0.50)
                    
                    multiplier_dict = {target_stage: 1 - total_efficiency}
                    res_opt = sim.run_simulation(resource_multiplier=multiplier_dict)
                    
                    mean_base = st.session_state.base_results['Total_Duration'].mean()
                    mean_opt = res_opt['Total_Duration'].mean()
                    time_saved = mean_base - mean_opt
                    
                    stage_opt_duration = res_opt[target_stage].mean()
                    total_resource_cost = res_params["cost_per_day"] * qty * stage_opt_duration
                    
                    def calc_expected_loss(durations, target, penalty):
                        failed = durations[durations > target]
                        avg_delay = failed.mean() - target if len(failed) > 0 else 0
                        prob_fail = len(failed) / len(durations)
                        return avg_delay * penalty * prob_fail
                        
                    loss_base = calc_expected_loss(st.session_state.base_results['Total_Duration'], eval_target, daily_penalty)
                    loss_opt = calc_expected_loss(res_opt['Total_Duration'], eval_target, daily_penalty)
                    
                    gross_savings = loss_base - loss_opt
                    net_benefit = gross_savings - total_resource_cost
                    roi = (net_benefit / total_resource_cost) * 100 if total_resource_cost > 0 else 0
                    
                    c_res1, c_res2 = st.columns([1, 1.5])
                    with c_res1:
                        if net_benefit > 0:
                            st.markdown(f"""
                            <div class="decision-box approved">
                                <h2>✅ LAYAK DIEKSEKUSI</h2>
                                <p>ROI: <b>+{roi:.1f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="decision-box rejected">
                                <h2>❌ DITOLAK</h2>
                                <p>Rugi Finansial: ROI Negatif</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with c_res2:
                        st.markdown(f"""
                        <table style="width:100%; font-size:1.05rem; background:white; border-radius:8px; border:1px solid #e2e8f0; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding:10px;">📉 Waktu Dihemat</td><td style="padding:10px; font-weight:bold; text-align:right;">{time_saved:.1f} Hari</td></tr>
                            <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding:10px;">📈 Denda Dihindari</td><td style="padding:10px; font-weight:bold; color:#10b981; text-align:right;">+ Rp {gross_savings/1e6:,.1f} Jt</td></tr>
                            <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding:10px;">💰 Biaya Resource Extra</td><td style="padding:10px; font-weight:bold; color:#ef4444; text-align:right;">- Rp {total_resource_cost/1e6:,.1f} Jt</td></tr>
                            <tr style="background:#f8fafc;"><td style="padding:12px; font-weight:bold;">NILAI BERSIH (NET BENEFIT)</td><td style="padding:12px; font-weight:bold; font-size:1.2rem; text-align:right; color:{'#10b981' if net_benefit > 0 else '#ef4444'};">Rp {net_benefit/1e6:,.1f} Jt</td></tr>
                        </table>
                        """, unsafe_allow_html=True)

    else:
        st.info("👈 Silakan tekan 'JALANKAN ANALISIS' di panel sebelah kiri untuk memulai komputasi model.")

if __name__ == "__main__":
    main()