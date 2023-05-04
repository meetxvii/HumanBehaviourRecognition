import streamlit as st


st.title("Login")

if st.button("Login as admin"):
    st.session_state.isLoggedIn = True
    st.experimental_rerun()

    
with st.form("Login"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Submit")

    if submitted:
        if username == "admin" and password == "admin":
            st.success("Logged in as admin")
            st.session_state.isLoggedIn = True
        else:
            st.error("The username or password you entered is incorrect")
