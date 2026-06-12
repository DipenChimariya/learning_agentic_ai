import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="AI Medical Analysis Dashboard", layout="wide")
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.2)


st.title("🩺Blood Work Analyzer")
st.write("Upload or paste a patient's raw text blood report to extract insights and generate a customized Nepali diet plan.")
st.markdown("---")


col1, col2 = st.columns([1, 1.2], gap="large")

# --- LEFT COLUMN: INPUT ---
with col1:
    st.subheader("📋 Input Patient Data")
    
    # Text area for pasting raw blood report data
    blood_report = st.text_area(
        "Paste Raw Blood Report Text Here:",
        height=500,
        placeholder="Example:\nGlucose Fasting: 140 mg/dL (Reference: 70-100)\nVitamin D3: 12 ng/mL (Reference: 30-100)"
    )
    
    # Clickable Button
    analyze_button = st.button("🚀 Analyze Report & Generate Diet", type="primary", use_container_width=True)

# --- RIGHT COLUMN: AI ANALYSIS OUTPUT ---
with col2:
    st.subheader("📊 AI Analysis Results")
    
    # Only run the chain when the user clicks the button and has provided input
    if analyze_button:
        if not blood_report.strip():
            st.error("Please paste a valid blood report text first!")
        else:
            with st.spinner("Processing medical data and calculating nutrition plan..."):
                try:
                    # ---- STAGE 1: EXTRACTION ----
                    extraction_prompt = f"""
                    You are a medical data extraction assistant.
                    From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL 
                    based on the reference ranges provided in the report.
                    Format your response as:
                    - Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range
                    
                    Blood Report:
                    {blood_report}
                    """
                    
                   
                    extracted_values = llm.invoke(extraction_prompt).text
                    with st.expander("✅ View Extracted Test Values (Stage 1)", expanded=False):
                        st.code(extracted_values, language="text")
                    
                    # ---- STAGE 2: DIET & SUMMARY ----
                    diet_prompt = f"""
                    You are a clinical nutritionist specializing in Nepali dietary habits.
                    Based on the blood work analysis below, write:
                    1. A short health summary in 4-5 lines explaining the patient's condition in simple language
                    2. A short, practical Nepali diet plan having only two sections:
                       ### (1) Foods to avoid
                       ### (2) Foods to eat more of
                    Do not include any other sections in the diet plan.
                    
                    Blood Work Analysis:
                    {extracted_values}
                    """
                    
                    diet_response = llm.invoke(diet_prompt).text
                    
                    # ---- STAGE 3: SPLITTING THE OUTPUT FOR CLEAN UI ----
                    # We can use standard string splitting or let the AI structure it. 
                    # For a clean visual layout, let's display the output in callout blocks:
                    
                    st.success("Analysis Complete!")
                    st.markdown("### 📝 Patient Health Summary")
                    st.info(diet_response) # Displays the generated insights clearly
                    
                except Exception as e:
                    st.error(f"An error occurred during parsing: {e}")
                    
    else:
        # Placeholder state before user clicks the button
        st.info("Waiting for blood report data input... Click the analysis button on the left to activate.")