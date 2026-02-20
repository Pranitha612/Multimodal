import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os
import sys

from utils.detection import PersonDetector, FaceDetector
from models.emotion_model import EmotionRecognizer
from models.relationship_model import RelationshipDetector
from utils.nlp_generator import generate_advanced_description

st.set_page_config(
    page_title="Emotion & Relationship Recognition",
    page_icon="😊",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all models"""
    st.info("Loading detection models...")
    person_detector = PersonDetector()
    face_detector = FaceDetector()
    
    emotion_model_path = 'checkpoints/emotion_model_best.pth'
    relationship_model_path = 'checkpoints/relationship_model_best.pth'
    
    st.info("Loading emotion recognition model...")
    emotion_recognizer = EmotionRecognizer(
        emotion_model_path if os.path.exists(emotion_model_path) else None
    )
    
    st.info("Loading relationship detection model...")
    relationship_detector = RelationshipDetector(
        relationship_model_path if os.path.exists(relationship_model_path) else None
    )
    
    st.success("All models loaded successfully!")
    
    return person_detector, face_detector, emotion_recognizer, relationship_detector

def process_image(image, person_detector, face_detector, emotion_recognizer, relationship_detector):
    """Process uploaded image and generate results"""
    
    temp_path = 'temp_image.jpg'
    cv2.imwrite(temp_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    
    with st.spinner("Detecting persons..."):
        persons = person_detector.detect_persons(temp_path)
    
    if not persons:
        os.remove(temp_path)
        return None, "No persons detected in the image."
    
    st.success(f"✓ Detected {len(persons)} person(s)")
    
    with st.spinner("Extracting faces..."):
        faces = face_detector.extract_faces(np.array(image), persons)
    
    if not faces:
        os.remove(temp_path)
        return None, "No faces detected in the image."
    
    st.success(f"✓ Extracted {len(faces)} face(s)")
    
    with st.spinner("Recognizing emotions..."):
        emotions = []
        face_data_with_embeddings = [] # store all data needed for relationship detection
        
        for face_data in faces:
            # Update: Get features + prediction
            result = emotion_recognizer.predict(face_data['image'], return_features=True)
            emotions.append({
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities']
            })
            
            # Enrich face_data with embedding for relationship detection
            face_data_copy = face_data.copy()
            face_data_copy['embedding'] = result['features']
            face_data_with_embeddings.append(face_data_copy)
    
    st.success(f"✓ Recognized emotions for {len(emotions)} person(s)")
    
    with st.spinner("Detecting relationships..."):
        # Pass data with embeddings
        relationships = relationship_detector.predict(face_data_with_embeddings)
    
    st.success(f"✓ Detected {len(relationships)} relationship(s)")
    
    with st.spinner("Generating description..."):
        description = generate_advanced_description(emotions, relationships)
    
    output_image = np.array(image).copy()
    
    for i, (face_data, emotion_data) in enumerate(zip(faces, emotions)):
        x1, y1, x2, y2 = map(int, face_data['bbox'])
        
        color = (0, 255, 0)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
        
        label = f"P{i+1}: {emotion_data['emotion']} ({emotion_data['confidence']:.2f})"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
        cv2.putText(output_image, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    results = {
        'num_persons': len(persons),
        'emotions': emotions,
        'relationships': relationships,
        'description': description
    }
    
    os.remove(temp_path)
    
    return output_image, results

def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100;300;400;700&family=Inter:wght@200;300;400;500;700&display=swap');

        /* Core Global Canvas */
        .stApp {
            background-color: #020202 !important;
            background-image: 
                radial-gradient(circle at 50% 10%, rgba(30,30,50,0.4) 0%, transparent 40%),
                radial-gradient(circle at 10% 90%, rgba(10,20,30,0.4) 0%, transparent 50%),
                repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px) !important;
            font-family: 'Inter', sans-serif !important;
            color: #d1d5db !important;
        }

        /* Cleanest top bar */
        header { background-color: transparent !important; }
        .st-emotion-cache-1avcm0n { visibility: hidden; } /* hide deploy/st default top right menu */

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 200 !important;
            letter-spacing: -0.05em !important;
            color: #ffffff !important;
        }

        /* Typography - Subheaders override */
        h2 { font-size: 1.8rem !important; letter-spacing: -0.02em !important; border-bottom: 1px solid rgba(255,255,255,0.05) !important; padding-bottom: 0.8rem !important; margin-bottom: 1.5rem !important; }
        h3 { font-size: 1.2rem !important; color: #9ca3af !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 500 !important; }

        /* Sidebar - The Control Panel */
        [data-testid="stSidebar"] {
            background: rgba(5, 5, 5, 0.85) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255,255,255,0.03) !important;
        }
        [data-testid="stSidebar"] .stMarkdown { color: #a1a1aa !important; }

        /* Uplpad Zone - Scanning Interface */
        [data-testid="stFileUploadDropzone"] {
            background: rgba(10, 10, 12, 0.5) !important;
            border: 1px dashed rgba(255, 255, 255, 0.1) !important;
            border-radius: 4px !important;
            padding: 3rem !important;
            transition: all 0.5s ease-in-out !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: rgba(56, 189, 248, 0.6) !important;
            background: rgba(15, 20, 30, 0.7) !important;
            box-shadow: inset 0 0 30px rgba(56, 189, 248, 0.05), 0 0 20px rgba(56, 189, 248, 0.1) !important;
        }
        .stFileUploader > div > div > small { display: none !important; } /* hide default instructions */

        /* Buttons - Execution Triggers */
        .stButton > button {
            background: transparent !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 2px !important;
            padding: 1rem 3rem !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 400 !important;
            font-size: 0.9rem !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            position: relative;
            overflow: hidden;
        }
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.5s ease;
        }
        .stButton > button:hover::before { left: 100%; }
        .stButton > button:hover {
            border-color: #38bdf8 !important;
            color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.05) !important;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.2), inset 0 0 10px rgba(56, 189, 248, 0.1) !important;
        }

        /* Metrics Data Fields */
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: #f8fafc !important;
            font-size: 3.5rem !important;
            font-weight: 100 !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 300;
        }

        /* Image Display */
        [data-testid="stImage"] img {
            border-radius: 2px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            filter: contrast(1.05) brightness(0.95);
            transition: filter 0.8s ease;
        }
        [data-testid="stImage"]:hover img {
            filter: contrast(1.1) brightness(1.05);
            box-shadow: 0 0 40px rgba(255, 255, 255, 0.02);
        }

        /* Divider lines */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
            margin: 3rem 0 !important;
        }

        /* Expander / Analysis Readouts */
        .streamlit-expanderHeader {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: 2px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
            letter-spacing: 1px !important;
            color: #94a3b8 !important;
            transition: border-color 0.3s ease !important;
        }
        .streamlit-expanderHeader:hover {
            border-color: rgba(255,255,255,0.2) !important;
            color: #ffffff !important;
        }
        .streamlit-expanderContent {
            border: 1px solid rgba(255,255,255,0.05) !important;
            border-top: none !important;
            background: rgba(5,5,5,0.5) !important;
            padding: 1.5rem !important;
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background-color: #38bdf8 !important;
            border-radius: 0 !important;
        }
        .stProgress > div > div {
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 0 !important;
        }

        /* Notification boxes (Success/Info) */
        div.stAlert {
            background: rgba(10, 10, 12, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-left: 2px solid rgba(255, 255, 255, 0.3) !important;
            color: #a1a1aa !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
            border-radius: 0 !important;
        }
        div.stAlert[data-baseweb="notification"] > div[role="alert"] { color: #a1a1aa !important; }

        /* The Custom Description Console */
        .console-box {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255,255,255,0.05);
            border-left: 2px solid #38bdf8;
            padding: 2rem;
            font-family: 'Inter', sans-serif;
            font-weight: 300;
            font-size: 1.1rem;
            line-height: 1.8;
            color: #e2e8f0;
            margin: 2rem 0;
            position: relative;
            animation: textReveal 1.2s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
        }
        .console-box::before {
            content: 'SYS.OUTPUT //';
            position: absolute;
            top: -12px;
            left: 15px;
            background: #020202;
            padding: 0 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #64748b;
            letter-spacing: 2px;
        }

        /* Custom Cinematic Intro Animation */
        @keyframes initFade {
            0% { opacity: 0; filter: blur(10px); transform: translateY(20px); }
            100% { opacity: 1; filter: blur(0px); transform: translateY(0); }
        }
        .cinematic-header {
            animation: initFade 1.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            text-align: center;
            padding: 5rem 0 3rem 0;
        }
        .cinematic-header-sub {
            font-family: 'JetBrains Mono', monospace;
            color: #38bdf8;
            font-size: 0.75rem;
            letter-spacing: 0.4em;
            text-transform: uppercase;
            margin-bottom: 1.5rem;
            opacity: 0.8;
        }
        .cinematic-header-main {
            font-size: 4.5rem;
            font-weight: 200;
            letter-spacing: -3px;
            color: #ffffff;
            line-height: 1;
            margin-bottom: 2rem;
            text-shadow: 0 0 40px rgba(255,255,255,0.1);
        }
        .cinematic-header-desc {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #64748b;
            max-width: 500px;
            margin: 0 auto;
            line-height: 1.8;
            letter-spacing: 0.05em;
        }

        @keyframes textReveal {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Hide the top right toolbar links */
        [data-testid="stDecoration"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="cinematic-header">
            <div class="cinematic-header-sub">// COGNITIVE VISION MODULE</div>
            <div class="cinematic-header-main">Neural Interface</div>
            <div class="cinematic-header-desc">
                Deploying spatial emotion dynamics and relationship inference protocols. Enter visual data stream to initiate sequential analysis.
            </div>
        </div>
        <hr style="margin-bottom:4rem !important; opacity:0.3;">
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### DATA STREAM")
    st.sidebar.info(
        """
        **PROTOCOL: OMEGA**
        
        SYSTEM CAPABILITIES:
        [x] Multi-Subject Detection  
        [x] Micro-Expression Parsing  
        [x] Relational Inference Engine  
        [x] NLP Synthesis
        
        STACK: YOLOv8 / MTCNN / ResNet18
        """
    )
    
    st.sidebar.markdown("### SYSTEM STATUS")
    
    if torch.cuda.is_available():
        st.sidebar.success(f"GPU ONLINE: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("WARNING: NEURO-PROCESSOR DETECTED CPU ONLY")
    
    with st.spinner("Loading AI models..."):
        person_detector, face_detector, emotion_recognizer, relationship_detector = load_models()
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📁 Choose an image file (JPG, PNG)", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear group photo with visible faces"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            
            with col2:
                st.subheader("🎯 Analysis In Progress")
                
                output_image, results = process_image(
                    image, person_detector, face_detector, 
                    emotion_recognizer, relationship_detector
                )
            
            if output_image is not None:
                with col2:
                    st.subheader("✨ Analysis Result")
                    st.image(output_image, use_container_width=True)
                
                st.divider()
                st.info("ANALYSIS COMPLETE")
                
                st.subheader("SYNTHESIS LOG")
                st.markdown(f'<div class="console-box">{results["description"]}</div>', unsafe_allow_html=True)
                
                st.divider()
                
                st.subheader("📊 Detailed Analysis")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("👥 People Detected", results['num_persons'])
                    
                    st.write("**😊 Emotions Detected:**")
                    for i, emotion_data in enumerate(results['emotions'], 1):
                        confidence_pct = emotion_data['confidence'] * 100
                        st.write(f"**Person {i}:** {emotion_data['emotion'].title()} "
                               f"({confidence_pct:.1f}% confidence)")
                
                with col_b:
                    st.metric("🤝 Relationships Found", len(results['relationships']))
                    
                    if results['relationships']:
                        st.write("**👫 Detected Relationships:**")
                        for rel in results['relationships']:
                            st.write(f"**Person {rel['person1']+1} & Person {rel['person2']+1}:** "
                                   f"{rel['relationship'].title()}")
                    else:
                        st.info("No specific relationships detected between individuals.")
                
                st.divider()
                
                with st.expander("🔬 View Emotion Probabilities"):
                    for i, emotion_data in enumerate(results['emotions'], 1):
                        st.write(f"**Person {i} - All Emotions:**")
                        probs = emotion_data['all_probabilities']
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        for emotion, prob in sorted_probs:
                            st.progress(prob, text=f"{emotion.title()}: {prob*100:.1f}%")
                        st.divider()
                
            else:
                st.error(results)
    
    else:
        st.markdown("<div style='text-align: center; border: 1px dashed rgba(255,255,255,0.1); padding: 4rem; background: rgba(5,5,5,0.5);'><h3 style='color:#a1a1aa; margin-bottom: 0.5rem;'>WAITING FOR VISUAL DATASTREAM</h3><p style='color:#52525b; font-family:\"JetBrains Mono\", monospace; font-size:0.8rem; letter-spacing:1px; margin:0;'>DRAG AND DROP OR BROWSE TO INITIATE SEQUENCE</p></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='opacity: 0.6; margin-top: 3rem; font-family:\"JetBrains Mono\", monospace; font-size: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("### DIAGNOSTIC PROTOCOLS:")
        st.markdown("""
        > FOR OPTIMAL EXTRACTION: Ensure high lux parameters  
        > REQUIREMENT: Subjects should not be occluded  
        > CALIBRATION: Engine optimized for 2-10 subjects   
        > EXCLUDE: High kinetic motion, low-res capture sources
        """)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()