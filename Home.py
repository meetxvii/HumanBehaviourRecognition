import streamlit as st
import os
import shutil
from streamlit_option_menu import option_menu
import pyrebase
import datetime
import extra_streamlit_components as stx
import os
from supabase import create_client, Client

url = "https://fzpvefiyomazymkowund.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ6cHZlZml5b21henlta293dW5kIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODI4NTYyNjQsImV4cCI6MTk5ODQzMjI2NH0.qxmzEFCys_1i1Auln9xO1GRMpMYlznSo-qNjbLYK6cI"
supabase: Client = create_client(url, key)



try: 
    user = supabase.auth.get_user()
except:
    user = None

page_placeholder = st.empty()
# with st.sidebar:
#     menu_placeholder.empty()
#     with menu_placeholder:
#         menu_placeholder.empty()
#         if user is not None:
#             page = option_menu("Menu", ["Home", "Dashboard", "Logout"])
#         else:
#             page = option_menu("Menu", ["Home", "Login", "Signup"])

with st.sidebar:
    if user is not None:
        page = option_menu("Menu", ["Home", "Login", "Signup"],key="menu")

st.write(page)

if page == "Home":
    
    with page_placeholder:
        st.write("# Home")


if page == "Login":
    
    with page_placeholder:    
        with st.form("Login"):
            st.write("# Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Login")

            if submit_button:
                try:
                    user = supabase.auth.sign_in_with_password({"email": email, "password": password})      
                    
                    with st.sidebar:
                        page = option_menu("Menu", ["Home", "Dashboard", "Logout"])
                        page_placeholder.empty()
                        
                except Exception as e: 
                    st.error("Invalid email or password")
                    st.write(e)
            

if page == "Signup":
    page_placeholder.empty()
    with page_placeholder:
        st.title("Signup")
        with st.form("Signup"):
            st.write("# Signup")
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Signup")

            if submit_button:
                try:
                    res = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                        })
                    user = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password,
                        })

                    

                    st.success("Signup successful")
                    # with menu_placeholder:
                    #     menu_placeholder.empty()
                    #     page = option_menu("Menu", ["Home", "Login", "Signup"])
                    #     page_placeholder.empty()
                except Exception as e:

                    st.error("Signup failed")
                    st.write(e)
                    st.write("If you already have an account, please login")
                

if page == "Logout":
    page_placeholder.empty()
    
                