def inject_branding():
    st.markdown(
        f"""
        <style>
            /* Main background */
            .stApp {{
                background-color: #FAFBFC;
                color: #111827 !important;
            }}
            
            /* Chat Messages Styling */
            .stChatMessage, .stChatMessage p, .stChatMessage div, [data-testid="stChatMessageContent"] {{
                color: #111827 !important;
            }}

            [data-testid="stChatMessage"] {{
                background-color: #F3F4F6 !important;
                border-radius: 10px;
            }}

            /* Chat Input Box Text & Placeholder Styling */
            [data-testid="stChatInput"] textarea {{
                color: #FFFFFF !important;
            }}
            
            [data-testid="stChatInput"] textarea::placeholder {{
                color: #9CA3AF !important;
            }}
            
            /* Sidebar Styling - Dark Theme */
            [data-testid="stSidebar"] {{
                background-color: #111827 !important;
                border-right: 1px solid #1F2937 !important;
            }}
            
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] span {{
                color: #FFFFFF !important;
            }}
            
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
            [data-testid="stSidebar"] p {{
                color: #9CA3AF !important;
            }}

            /* Header Styling */
            .app-header {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 1rem 1.25rem;
                background-color: {ACCENT_COLOR};
                border-radius: 10px;
                margin-bottom: 1.25rem;
            }}
            .app-header h1 {{
                color: white !important;
                font-size: 1.4rem;
                margin: 0;
                font-weight: 600;
            }}
            .app-header p {{
                color: #DCE8F5 !important;
                margin: 0;
                font-size: 0.85rem;
            }}
            .source-pill {{
                display: inline-block;
                background-color: {ACCENT_LIGHT};
                color: {ACCENT_COLOR};
                border: 1px solid #B9D0E6;
                border-radius: 999px;
                padding: 2px 10px;
                margin: 2px 4px 0 0;
                font-size: 0.75rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="app-header">
            <div>🎓</div>
            <div>
                <h1>{SCHOOL_NAME}</h1>
                <p>Ask about grading, attendance, and other school policies</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
