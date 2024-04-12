import streamlit as st
import pandas as pd
from datetime import datetime
import SMjournal as smj
import plotly.express as px

#change name of stats

# smjournal dont show option
# change beta
# correlation thresh

st.set_page_config(page_title="Stats",page_icon=None)

def select_cols(cols):
  dims=["Geographies","Weeks"]
  weight=[i for i in cols if ".LEQHHwgt" in i][0]
  y_col=[i for i in cols if ".LEQHHDEP" in i][0]
  x_cols=[i for i in cols if i not in dims+[weight,y_col]]
  return dims, x_cols, y_col, weight

def _max_width_(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )
_max_width_(90)



st.title('Stats')
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Collinearity Tests", "Feature Selection", 'Meta Regression'])

with tab1:
   
    holder = st.empty()
    uploaded_file = holder.file_uploader("Choose a CSV to upload, make sure to have Weeks, Geographies, the dependent variable and weight")
    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        df['Weeks']=pd.to_datetime(df['Weeks'])
        df=df.sort_values('Weeks')
     

        st.text("")
        st.text("")
        geos = st.multiselect(
                'Select the Geographies to use',
                list(df['Geographies'].unique()),
                list(df['Geographies'].unique()))
        first_day=df['Weeks'].unique()[0]
        last_day=df['Weeks'].unique()[-1]
   
       
        weeks = st.slider(
            "Select dates.",
            value=(first_day.to_pydatetime(),last_day.to_pydatetime())
        )
        

        df=df.loc[df['Geographies'].isin(geos)]
        df=df.loc[(df['Weeks']>weeks[0]) & (df['Weeks']<weeks[1])]
        st.write(df)
        holder.empty()

        
with tab2:

    if uploaded_file is not None:
        dims, x_cols, y_col, weight=select_cols(df.columns)
        selected = st.multiselect(
                'Select the Columns to use',
                x_cols,
               [i for i in x_cols if ('.hol' not in i.lower()) and ('.mkt' not in i.lower()) and ('.dummy' not in i.lower())])
        filt = st.number_input('Insert a Correlation filter')
        if st.button('Understand Collinearity'):

            clus, fig, plt, corrplot=smj.collinearity_test(df[['Weeks']+selected],streamlit=st,filter=filt)
            st.pyplot(corrplot)
            st.plotly_chart(fig, use_container_width=True)
            st.pyplot(plt)

    
with tab3:
    if uploaded_file is not None:
        control_cols = st.multiselect(
                'Select the Control Columns to use',
                x_cols,
               [i for i in x_cols if ('.hol' not in i.lower()) and ('.mkt' not in i.lower()) and ('.dummy' not in i.lower())])
        const_cols = st.multiselect(
                'Select the Constant Columns to use',
                x_cols,
               [i for i in x_cols if ('.hol' in i.lower()) or ('.mkt' in i.lower()) or ('.dummy' in i.lower())])
        if st.button('Select Features'):
            dims, x_cols, y_col, weight=select_cols(df.columns)
            fig2, plot0, plot1, toploten0, toploten1=smj.importance_test(df[[i for i in df.columns if "Geographies" not in i]], y_col, weight_col=None, control_cols=control_cols, const_cols=const_cols,streamlit=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.subheader('Linear Selection')
        
            st.plotly_chart(px.bar(toploten0.loc[toploten0['imp']!=0], x="imp", y="cols", orientation='h'), use_container_width=True)
            # st.pyplot(plot0.get_figure())
            st.subheader('Non Linear Selection')
            st.plotly_chart(px.bar(toploten1.loc[toploten1['imp']!=0], x="imp", y="cols", orientation='h'), use_container_width=True)
            # st.write(toploten1)
            # st.pyplot(plot1.get_figure())

with tab4:
    st.title("Collinearity Tests")

   



    





