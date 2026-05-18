import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import json

# Configuração da página do Streamlit
st.set_page_config(page_title="Detetor de Peças de Lego - IA", layout="wide")
st.title("🏭 Sistema Industrial de Deteção de Peças de Lego")
st.write("Carregue uma imagem ou use a câmara para identificar os componentes de construção.")

# ==========================================
# REQUISITO: SELEÇÃO DO MODELO (Dropdown)
# ==========================================
st.sidebar.header("Configurações do Modelo")
modelo_selecionado = st.sidebar.selectbox(
    "Escolha o modelo YOLOv8:",
    ["YOLOv8 Nano (Rápido)", "YOLOv8 Small (Preciso)"]
)

# Mapeamento para os caminhos reais dos vossos ficheiros .pt
caminho_modelo = "modelos/yolov8_lego/weights.pt" # Ajustem conforme o modelo selecionado

@st.cache_resource
def carregar_modelo(path):
    return YOLO(path)

try:
    model = carregar_modelo(caminho_modelo)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}. Certifique-se de que o ficheiro .pt está na pasta correta.")

# ==========================================
# BÓNUS: SLIDER DE CONFIANÇA
# ==========================================
conf_threshold = st.sidebar.slider("Limiar de Confiança (Threshold)", 0.0, 1.0, 0.5, 0.05)

# ==========================================
# REQUISITO: INPUT DE IMAGEM / VÍDEO
# ==========================================
metodo_input = st.radio("Selecione o método de entrada:", ["Upload de Imagem", "Câmara em Tempo Real"])

if metodo_input == "Upload de Imagem":
    uploaded_file = st.file_uploader("Escolha uma imagem de peças de Lego...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Converter o arquivo carregado para o formato OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        imagem_bgr = cv2.imdecode(file_bytes, 1)
        
        # Executar a inferência do YOLO
        resultados = model.predict(source=imagem_bgr, conf=conf_threshold)
        
        # O YOLO já lida com o desenho das caixas através do método .plot()
        imagem_desenhada = resultados.plot()
        # Converter BGR (OpenCV) para RGB (Streamlit)
        imagem_rgb = cv2.cvtColor(imagem_desenhada, cv2.COLOR_BGR2RGB)
        
        # Apresentar a imagem com as Bounding Boxes
        st.image(imagem_rgb, caption="Resultado da Deteção", use_container_width=True)
        
        # ==========================================
        # REQUISITO: VISTA DE SAÍDA ESTRUTURADA
        # ==========================================
        st.subheader("📊 Dados Estruturados da Inferência")
        
        dados_detecao = []
        # Extrair caixas, classes e confianças
        for box in resultados.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()
            conf = float(box.conf)
            cls_id = int(box.cls)
            nome_classe = model.names[cls_id]
            
            dados_detecao.append({
                "Peça (Classe)": nome_classe,
                "Confiança": f"{conf*100:.2f}%",
                "Coordenadas Box [x1, y1, x2, y2]": [round(x1), round(y1), round(x2), round(y2)]
            })
            
        if dados_detecao:
            # Apresentar em formato de Tabela
            df = pd.DataFrame(dados_detecao)
            st.dataframe(df, use_container_width=True)
            
            # BÓNUS: Exportação dos resultados em JSON
            st.download_button(
                label="📥 Exportar Resultados (JSON)",
                data=json.dumps(dados_detecao, indent=4),
                file_name="resultados_lego.json",
                mime="application/json"
            )
        else:
            st.info("Nenhuma peça de Lego detetada com o limiar de confiança atual.")

elif metodo_input == "Câmara em Tempo Real":
    st.warning("Dica: Para o Streamlit em produção local, o 'st.camera_input' captura fotos individuais. Para vídeo contínuo em tempo real, usa-se o OpenCV puro num loop local, ou o componente 'streamlit-webrtc'.")
    
    img_cam = st.camera_input("Tire uma foto da bancada de Lego")
    if img_cam is not None:
        bytes_data = img_cam.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        resultados = model.predict(source=cv2_img, conf=conf_threshold)
        imagem_rgb = cv2.cvtColor(resultados.plot(), cv2.COLOR_BGR2RGB)
        
        st.image(imagem_rgb, caption="Deteção da Câmara", use_container_width=True)
        # Podem repetir a lógica da tabela estruturada aqui para a foto da câmara