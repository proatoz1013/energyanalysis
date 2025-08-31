import streamlit as st

def main():
    st.set_page_config(page_title="Test Tab Example", layout="wide")
    st.title("Streamlit Test Tab Demo")

    tabs = st.tabs(["Test Tab"])
    with tabs[0]:
        st.header("This is the Test Tab")
        st.write("Welcome to your new test tab!")
        st.write("Add your custom content here.")

if __name__ == "__main__":
    main()
