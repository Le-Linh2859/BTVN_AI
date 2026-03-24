import streamlit as st
import pandas as pd
import joblib as jb

# =========================
# Load model
# =========================
model = jb.load("catboost_model.pkl")

st.set_page_config(
    page_title="Cảnh báo học vụ cho sinh viên",
    layout="wide"
)

st.title("🎓 Hệ thống dự đoán Cảnh báo học vụ cho sinh viên")

st.write(
"""
Ứng dụng sử dụng mô hình Machine Learning để dự đoán nguy cơ học tập của sinh viên.

Kết quả:
- 0 → Normal
- 1 → Academic Warning
- 2 → Dropout
"""
)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "Thông tin cơ bản",
    "Thông tin cá nhân",
    "Điểm danh môn học"
])

# =========================
# TAB 1
# =========================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Tuổi", 16, 40, 20)

        tuition_debt = st.number_input(
            "Số tiền học phí còn nợ",
            0.0, 100000000.0, 0.0
        )

    with col2:
        count_f = st.number_input(
            "Số môn điểm F",
            0, 20, 0
        )

        training_score = st.slider(
            "Điểm rèn luyện",
            0, 100, 70
        )

# =========================
# TAB 2
# =========================
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Giới tính",
            ["Male", "Female"]
        )

        hometown = st.text_input(
            "Quê quán"
        )

        admission_mode = st.selectbox(
            "Hình thức tuyển sinh",
            ["Exam", "Direct", "Other"]
        )

    with col2:
        current_address = st.text_input(
            "Địa chỉ hiện tại"
        )

        english_level = st.selectbox(
            "Trình độ tiếng Anh",
            ["A1", "A2", "B1", "B2", "C1"]
        )

        club_member = st.selectbox(
            "Tham gia câu lạc bộ",
            ["Yes", "No"]
        )

    st.subheader("Thông tin văn bản")

    advisor_notes = st.text_area(
        "Nhận xét của cố vấn học tập"
    )

    personal_essay = st.text_area(
        "Bài tự đánh giá của sinh viên"
    )

# =========================
# TAB 3
# =========================
with tab3:

    st.write("Nhập tỷ lệ tham gia lớp học (%)")

    attendance = {}

    for i in range(1, 41):

        subject = f"Att_Subject_{str(i).zfill(2)}"

        attendance[subject] = st.number_input(
            f"Môn {i}",
            0.0,
            10.0,
            0.0
        )

# =========================
# PREDICT BUTTON
# =========================
st.divider()

if st.button("🔎 Dự đoán trạng thái học tập", use_container_width=True):

    data = {
        "Age": age,
        "Tuition_Debt": tuition_debt,
        "Count_F": count_f,
        "Training_Score_Mixed": training_score,
        "Gender": gender,
        "Hometown": hometown,
        "Current_Address": current_address,
        "Admission_Mode": admission_mode,
        "English_Level": english_level,
        "Club_Member": club_member,
        "Advisor_Notes": advisor_notes,
        "Personal_Essay": personal_essay
    }

    data.update(attendance)

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]

    st.subheader("Kết quả dự đoán")

    if pred == 0:
        st.success("Sinh viên thuộc nhóm: NORMAL")

    elif pred == 1:
        st.warning("Sinh viên thuộc nhóm: ACADEMIC WARNING")

    else:
        st.error("Sinh viên thuộc nhóm: DROP OUT")
