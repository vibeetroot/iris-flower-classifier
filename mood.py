import streamlit as st

st.title("🌸 Little Mood Diary 🌸")
mood = st.selectbox("How are you feeling today?",["Happy 😊", "Sad 😔", "Tired 😴", "Excited 🤩", "Angry 😡"])
note = st.text_input("Write a little note about your day:")
if st.button("Save Entry 💌"):
        st.write("Your mood:", mood)
        st.write("Your note:", note)

if "Happy" in mood:
        st.success("Keep shining 🌞")
elif "Sad" in mood:
        st.info("Don't be 💙")
elif "Tired" in mood:
        st.warning("Get some rest 😴")
elif "Excited" in mood:
        st.success("Yayieeeeeeeeee ⚡")
elif "Angry" in mood:
        st.success("Ugh thats bad :(")
    