import streamlit as st

def run():
    st.title("Detect Image")



if __name__ == "__main__":
    if "isLoggedIn" not in st.session_state or st.session_state.isLoggedIn == False:
        st.error("You must login first!")
        st.stop()    
    run()