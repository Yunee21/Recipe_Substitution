# %%
import streamlit as st
import pandas as pd
import numpy as np

# %%
st.title("신장질환 맞춤 레시피 대체 시스템")

# -----------------------------
# 👥 신체 정보 입력 (3열 구성)
# -----------------------------
st.markdown("📏 신체 정보")
with st.expander("🧬 신체 정보", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.radio("성별", ["남성", "여성"], horizontal=True)
    with col2:
        height = st.text_input("신장(cm)", placeholder="예: 170")
    with col3:
        weight = st.text_input("체중(kg)", placeholder="예: 65")

# -----------------------------
# 🧬 신장질환 정보 입력
# -----------------------------
with st.expander("🧬 신장질환 정보", expanded=True):
    input_method = st.radio(
        "입력 방식을 선택하세요",
        ("신장질환 단계 선택", "eGFR 수치 입력")
    )

    kidney_stage = None
    kidney_dialysis = None
    egfr = None

    if input_method == "신장질환 단계 선택":
        kidney_stage = st.selectbox("현재 신장질환 단계를 선택하세요", ["1단계", "2단계", "3단계", "4단계", "5단계"])
        kidney_dialysis = st.selectbox("현재 투석 여부를 선택하세요", ["비투석", "복막투석", "혈액투석"])
    else:
        egfr = st.number_input("eGFR 수치 입력", min_value=0.0, max_value=200.0, step=0.1)
        if egfr >= 90:
            kidney_stage = "1단계"
        elif 60 <= egfr < 90:
            kidney_stage = "2단계"
        elif 30 <= egfr < 60:
            kidney_stage = "3단계"
        elif 15 <= egfr < 30:
            kidney_stage = "4단계"
        elif egfr < 15:
            kidney_stage = "5단계"

        kidney_dialysis = st.selectbox("현재 투석 여부를 선택하세요", ["비투석", "복막투석", "혈액투석"])

# %%
# -----------------------------
# 🧺 보유 식재료 입력
# -----------------------------
with st.expander("🧺 보유 식재료", expanded=True):
    ingredient_input = st.text_area(
        "현재 보유하고 있는 식재료를 입력하세요 (쉼표로 구분)",
        placeholder="예: 두부, 양파, 간장, 달걀, 시금치"
    )

    ingredient_list = []
    if ingredient_input:
        ingredient_list = [item.strip() for item in ingredient_input.split(",") if item.strip()]
        if ingredient_list:
            st.success("입력된 식재료 목록:")
            st.write(ingredient_list)
        else:
            st.info("보유 재료를 입력해주세요")

# %%
# -----------------------------
# 🍳 레시피 정보 입력
# -----------------------------
with st.expander("🍳 레시피 정보 업로드", expanded=True):
    uploaded_file = st.file_uploader("레시피 CSV 파일을 업로드하세요", type=["csv"])

    if uploaded_file is not None:
        import pandas as pd
        recipe_df = pd.read_csv(uploaded_file)

        st.success("레시피 파일 업로드가 완료되었습니다")
        
        # 데이터 확인
        st.subheader("📋 업로드된 레시피 데이터 미리보기")
        st.dataframe(recipe_df, use_container_width=True)

        # 선택적 기능: 사용자가 선택한 컬럼만 보기
        with st.expander("🔍 특정 컬럼만 보기"):
            selected_cols = st.multiselect("보고 싶은 컬럼 선택", recipe_df.columns.tolist(), default=recipe_df.columns.tolist())
            st.dataframe(recipe_df[selected_cols], use_container_width=True)
    else:
        st.info("섭취하고자 하는 레시피를 업로드해주세요")

# %%
# -----------------------------
# ✅ 제출 버튼 및 요약
# -----------------------------
st.markdown("---")

can_submit = (
    gender and height and weight
    and kidney_stage and kidney_dialysis
    and 'recipe_df' in locals()
)

if st.button("제출"):
    if can_submit:
        st.success("입력이 완료되었습니다")
        st.markdown("### ✅ 입력 요약")
        st.write(f"- 성별: {gender}")
        st.write(f"- 신장: {height} cm")
        st.write(f"- 체중: {weight} kg")
        st.write(f"- 신장질환 단계: {kidney_stage}")
        st.write(f"- 투석 여부: {kidney_dialysis}")
        if input_method == "eGFR 수치 입력":
            st.write(f"- eGFR 수치: {egfr}")
        if ingredient_list:
            st.write(f"- 보유 식재료: {', '.join(ingredient_list)}")
        st.write("✅ 레시피 파일 업로드 완료")
    else:
        st.error("필수 정보를 모두 입력하고, 레시피 파일을 업로드해야 제출할 수 있습니다")


