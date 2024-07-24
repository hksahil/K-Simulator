import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import duckdb
from streamlit_dynamic_filters import DynamicFilters
from matplotlib import style
from millify import millify

# Page Settings
st.set_page_config(page_title='Kellogg Dynamic Modelling Simulator',layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

if "load_state" not in st.session_state:
    st.session_state.load_state=False
# CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define df1
df_raw = pd.read_csv('Pre-Launched.csv')
df_raw = df_raw.dropna(subset=['launch_dt'])
df_raw = df_raw[df_raw['stage_gate']!='Hold']
df_raw['launch_dt']=pd.to_datetime(df_raw['launch_dt'])
df_raw.rename(columns = {'proj_nm':'Project','proj_id':'Proj_ID','geog_short':'Region','bu_short':'BU',
                         'prtflo_bkt':'Portfolio_Bucket','inc_nsv_yr_1':'insv_yr_1','inc_nsv_yr_2':'insv_yr_2',
                         'inc_nsv_yr_3':'insv_yr_3','proj_desc':'Project_Desc','launch_dt':'Launch_Date','launch_yr':'Launch_Year'},inplace=True)
raw_copy=copy.deepcopy(df_raw)
df_raw = df_raw[['fltr','Proj_ID','Project','Project_Desc','Launch_Date','Region','BU','Portfolio_Bucket',
                 'pd_or_ft_days','pkg_days','canblz_pct','gross_mrgn_pct',
                 'nsv_yr_1','nsv_yr_2','nsv_yr_3','insv_yr_1','insv_yr_2','insv_yr_3',
                 'gsv_yr_1','gsv_yr_2','gsv_yr_3']]
df_raw = df_raw[df_raw['fltr']==1]
df_raw = df_raw.drop(['fltr'], axis=1)
df_raw['Action']='Current'
df_raw_copy = copy.deepcopy(df_raw)
df_raw.set_index('Action', inplace=True)
df_raw.sort_values(by='Proj_ID',ascending=True,inplace=True)
df_raw_copy.sort_values(by='Proj_ID',ascending=False,inplace=True)

# To calculate 1 Year Rolling NSV/iNSV(s)
def calculate_rolling_values(row, doll_year, doll_wrap):
    if row['Launch Year'] == row['Year of NSV']:
        return row[doll_year]
    elif row['Launch Year'] == row['Year of NSV'] - 1:
        return row[doll_wrap]
    else:
        return 0

def weighted_average_of_group(df,dim_selected):
    df = df[df['Filter']==1]
    df['Weighted_Score1'] = df['Published Gross Margin Pct'] * df['Published NSV Year 1']
    df['Weighted_Score2'] = df['Simulated Gross Margin Pct'] * df['Simulated NSV Year 1']

    weighted_avg_df = df.groupby(dim_selected).apply(lambda x: pd.Series({
        'Published Gross Margin WA Pct': (x['Weighted_Score1'].sum() / x['Published NSV Year 1'].sum()),
        'Simulated Gross Margin WA Pct': (x['Weighted_Score2'].sum() / x['Simulated NSV Year 1'].sum())
    })).reset_index()
    
    return (weighted_avg_df)

def remove_decimal(number):
    if number.endswith('.0'):
        return number[:-2]
    else:
        return number

def Process_data(df, dfx):
    # Replace values based on condition
    
    columns_to_replace = ['Simulated NSV Year 1', 'Simulated NSV Year 2', 'Simulated NSV Year 3', 
                          'Simulated NSV Year 1 Rollup', 'Simulated NSV Year 3 Rollup', 
                          'Simulated NSV Year', 'Simulated NSV Wrap', 
                          'Simulated iNSV Year 1', 'Simulated iNSV Year 2', 'Simulated iNSV Year 3', 
                          'Simulated iNSV Year 1 Rollup', 'Simulated iNSV Year 3 Rollup', 
                          'Simulated iNSV Year', 'Simulated iNSV Wrap', 
                          'Simulated Canblz Pct', 'Simulated Gross Margin Pct', 'Simulated pd or ft Days','Simulated pkg Days',
                          'Simulated GSV Year 1','Simulated GSV Year 2','Simulated GSV Year 3','Simulated GSV Year 1 Rollup',
                          'Simulated GSV Year 3 Rollup','Simulated GSV Year','Simulated GSV Wrap']
    for column in columns_to_replace:
        df[column] = [row[column] if row['flag'] else None for i, row in df.iterrows()]

    # calculate R&D Days
    df['Simulated R&D Days'] = df['Simulated pd or ft Days'].fillna(0) + df['Simulated pkg Days'].fillna(0)
    
    # Convert 'yr_of_nsv' to string and remove decimal
    dfx['Simulated_yr_of_nsv'] = dfx['yr_of_nsv'].astype(str).apply(remove_decimal)
    df['Simulated_yr_of_nsv'] = df['Year of NSV'].astype(str).apply(remove_decimal)
    
    # Create 'Combined' column for joining
    df['Combined'] = df['Simulated_yr_of_nsv'] + "_" + df['Project']
    dfx['Combined'] = dfx['Simulated_yr_of_nsv'] + "_" + dfx['Project']
    
    df[['Portfolio_Bucket','BU','Region','Project_Desc']] = df[['Portfolio_Bucket','BU','Region','Project_Desc']].fillna('No Data')

    # Join DataFrames
    result_concat = df.join(dfx.set_index("Combined"), how='left', rsuffix="_df1", on="Combined")
    
    # Drop unnecessary columns and rename remaining columns
    columns_to_drop = ['Project_df1', 'Region_df1', 'Proj_ID_df1', 'yr_of_nsv', 'Launch_Date_df1', 
                       'Launch_Year', 'Simulated_yr_of_nsv', 'Simulated_yr_of_nsv_df1','launch_mth','fltr','nsv_yr_risk_adj','nsv_wrap_risk_adj','insv_yr_risk_adj','insv_wrap_risk_adj','gsv_yr_risk_adj',
              'gsv_wrap_risk_adj','BU_df1','food_catg_short','proj_type_short','brand_short','stage_gate_full','proj_yr','geog_nm','bu_nm','bus_catg_nm',
              'Project_Desc_df1','food_catg','mktg_or_scr_lead','proj_type','big_bets','tier','Portfolio_Bucket_df1','brand','priority_advatage_brand','mfg_loc',
              'stage_gate','project_id','insrt_dt','Combined']
    result_concat.drop(columns_to_drop, axis=1, inplace=True)

    result_concat.rename(columns={'nsv_yr': 'Published NSV Year', 'nsv_wrap': 'Published NSV Wrap', 
                                  'nsv_yr_1': 'Published NSV Year 1', 'nsv_yr_2': 'Published NSV Year 2', 
                                  'nsv_yr_3': 'Published NSV Year 3', 'nsv_yr_1_rollup': 'Published NSV Year 1 Rollup', 
                                  'nsv_yr_3_rollup': 'Published NSV Year 3 Rollup', 'insv_yr': 'Published iNSV Year', 
                                  'insv_wrap': 'Published iNSV Wrap', 'insv_yr_1': 'Published iNSV Year 1', 
                                  'insv_yr_2': 'Published iNSV Year 2', 'insv_yr_3': 'Published iNSV Year 3', 
                                  'insv_yr_1_rollup': 'Published iNSV Year 1 ROllup', 'insv_yr_3_rollup': 'Published iNSV Year 3 Rollup', 
                                  'rd_days': 'Published R&D Days', 'canblz_pct': 'Published Canblz Pct', 
                                  'gross_mrgn_pct': 'Published Gross Margin Pct','pd_or_ft_days':'Published pd or ft Days','pkg_days': 'Published pkg Days' ,
                                  'gsv_yr_1':'Published GSV Year 1','gsv_yr_2':'Published GSV Year 2','gsv_yr_3':'Published GSV Year 3',
                                  'gsv_yr_1_rollup':'Published GSV Year 1 Rollup','gsv_yr_3_rollup':'Published GSV Year 3 Rollup', 
                                  'gsv_yr':'Published GSV Year','gsv_wrap':'Published GSV Wrap',
                                  'Portfolio_Bucket':'Portfolio Bucket','Proj_ID':'Project ID','Project_Desc':'Project Desc','launch_dt':'Launch Date'
                                  }, inplace=True)
    
    # Adding 3 year rolling NSV 
    result_concat['Simulated 3 Year Rolling NSV']= result_concat['Simulated NSV Year'].fillna(0)+ result_concat['Simulated NSV Wrap'].fillna(0)
    result_concat['Published 3 Year Rolling NSV']= result_concat['Published NSV Year'].fillna(0)+ result_concat['Published NSV Wrap'].fillna(0)
    result_concat['Simulated 3 Year Rolling NSV'] = [None if row['flag'] ==False else row['Simulated 3 Year Rolling NSV'] for index, row in result_concat.iterrows()]
    # Adding 3 year rolling iNSV 
    result_concat['Simulated 3 Year Rolling iNSV']= result_concat['Simulated iNSV Year'].fillna(0)+ result_concat['Simulated iNSV Wrap'].fillna(0)
    result_concat['Published 3 Year Rolling iNSV']= result_concat['Published iNSV Year'].fillna(0)+ result_concat['Published iNSV Wrap'].fillna(0)
    result_concat['Simulated 3 Year Rolling iNSV'] = [None if row['flag'] ==False else row['Simulated 3 Year Rolling iNSV'] for index, row in result_concat.iterrows()]
    # Adding 3 year rolling GSV 
    result_concat['Simulated 3 Year Rolling GSV']= result_concat['Simulated GSV Year'].fillna(0)+ result_concat['Simulated GSV Wrap'].fillna(0)
    result_concat['Published 3 Year Rolling GSV']= result_concat['Published GSV Year'].fillna(0)+ result_concat['Published GSV Wrap'].fillna(0)
    result_concat['Simulated 3 Year Rolling GSV'] = [None if row['flag'] ==False else row['Simulated 3 Year Rolling GSV'] for index, row in result_concat.iterrows()]

    # Calculate rolling NSV
    result_concat['Published 1 Year Rolling NSV'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'Published NSV Year', 'Published NSV Wrap'), axis=1)
    result_concat['Simulated 1 Year Rolling NSV'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'Simulated NSV Year', 'Simulated NSV Wrap'), axis=1)
    result_concat['Simulated 1 Year Rolling NSV'] = result_concat['Simulated 1 Year Rolling NSV'].where(result_concat['flag'], None)

    # Calculate rolling iNSV
    result_concat['Published 1 Year Rolling iNSV'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'Published iNSV Year', 'Published iNSV Wrap'), axis=1)
    result_concat['Simulated 1 Year Rolling iNSV'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'Simulated iNSV Year', 'Simulated iNSV Wrap'), axis=1)
    result_concat['Simulated 1 Year Rolling iNSV'] = result_concat['Simulated 1 Year Rolling iNSV'].where(result_concat['flag'], None)
    
    #calculate rolling GSV
    result_concat['Published 1 Year Rolling GSV'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'Published GSV Year', 'Published GSV Wrap'), axis=1)
    result_concat['Simulated 1 Year Rolling GSV'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'Simulated GSV Year', 'Simulated GSV Wrap'), axis=1)
    result_concat['Simulated 1 Year Rolling GSV'] = result_concat['Simulated 1 Year Rolling GSV'].where(result_concat['flag'], None)

    # Calculate Count of Project
    sim_project_count = (result_concat['Project'].where(result_concat['flag'], None))
    publ_project_count = (result_concat['Project']).where(result_concat['Action']=='CURRENT',None)

    float_column_names = result_concat.select_dtypes(float).columns
    result_concat[float_column_names] = result_concat[float_column_names].fillna(0)

    # result_concat['Published 1 Year Rolling NSV']=result_concat['Published 1 Year Rolling NSV'].round(2)
    # result_concat['Simulated 1 Year Rolling NSV']=result_concat['Simulated 1 Year Rolling NSV'].round(2)

    string_column_names = result_concat.select_dtypes(object).columns
    result_concat[string_column_names] = result_concat[string_column_names].fillna('No Data')

    result_concat['Year of NSV'] = result_concat['Year of NSV'].astype(str)
    result_concat['Project ID'] = result_concat['Project ID'].astype(str)
    result_concat['Launch Year'] = result_concat['Launch Year'].astype(str)

    return result_concat,sim_project_count,publ_project_count

def sql_process(df):
    # Processing the table 
    z_df1 = duckdb.query("""
            select *  from ( 
                         
            with nsv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_nsv) over (partition by Project order by year_of_nsv ) next_mm_nsv
            from (
            select flag,upper(Action) as Action,Project,Proj_ID,Launch_Date,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
                launch_yr + launch_month_nov_dec_flag launch_year ,
                case
                    when launch_month_nov_dec_flag = 1 then 1
                    else launch_month
                end launch_month ,
                launch_yr + launch_month_nov_dec_flag +
                case
                    when mm_type = 'mm_nsv_yr_1' then 0
                    when mm_type = 'mm_nsv_yr_2' then 1
                    when mm_type = 'mm_nsv_yr_3' then 2
                    when mm_type = 'mm_nsv_yr_dummy' then 3
                    else 0
                end year_of_nsv ,
                mm_nsv *(13-launch_month) yearly_total ,
                mm_nsv *(launch_month-1) mm_nsv ,
                nsv_yr_1_rollup ,
                nsv_yr_3_rollup ,
                curr_yr
            from (             
                select flag,upper(Action) as Action ,Project,Proj_ID,Launch_Date,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
                    extract( year from cast(Launch_Date as date)) as launch_yr,
                    extract( month from cast(Launch_Date as date)) as launch_month ,
                    case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                    cast(coalesce(nsv_yr_1, 0) as decimal)/ 12 as mm_nsv_yr_1 ,
                    cast(coalesce(nsv_yr_2, 0) as decimal)/ 12 as mm_nsv_yr_2 ,
                    cast(coalesce(nsv_yr_3, 0) as decimal)/ 12 as mm_nsv_yr_3 ,
                    0 as mm_nsv_yr_dummy ,
                    cast(coalesce(nsv_yr_1, 0) as decimal) nsv_yr_1_rollup ,
                    cast(coalesce(nsv_yr_1, 0) as decimal) + cast(coalesce(nsv_yr_2, 0) as decimal) + cast(coalesce(nsv_yr_3, 0) as decimal) nsv_yr_3_rollup,
                    extract(year from current_date) curr_yr
                from
                    df ) unpivot (mm_nsv for mm_type in (mm_nsv_yr_1, mm_nsv_yr_2, mm_nsv_yr_3, mm_nsv_yr_dummy))) ),

			insv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_insv) over (partition by Project order by year_of_nsv ) next_mm_insv
            from (
            select BU,Project_Desc,Portfolio_Bucket,pd_or_ft_days,pkg_days,canblz_pct,gross_mrgn_pct
                ,flag,upper(Action) as Action,Project,Proj_ID,Launch_Date,Region,insv_yr_1,insv_yr_2,insv_yr_3,
                launch_yr + launch_month_nov_dec_flag launch_year ,
                case
                    when launch_month_nov_dec_flag = 1 then 1
                    else launch_month
                end launch_month ,
                launch_yr + launch_month_nov_dec_flag +
                case
                    when mm_type = 'mm_insv_yr_1' then 0
                    when mm_type = 'mm_insv_yr_2' then 1
                    when mm_type = 'mm_insv_yr_3' then 2
                    when mm_type = 'mm_insv_yr_dummy' then 3
                    else 0
                end year_of_nsv ,
                mm_insv *(13-launch_month) yearly_total ,
                mm_insv *(launch_month-1) mm_insv ,
                insv_yr_1_rollup ,
                insv_yr_3_rollup ,
                curr_yr
            from (             
                select BU,Project_Desc,Portfolio_Bucket,pd_or_ft_days,pkg_days,canblz_pct,gross_mrgn_pct
                    ,flag,upper(Action) as Action ,Project,Proj_ID,Launch_Date,Region,insv_yr_1,insv_yr_2,insv_yr_3,
                    extract( year from cast(Launch_Date as date)) as launch_yr,
                    extract( month from cast(Launch_Date as date)) as launch_month ,
                    case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                    cast(coalesce(insv_yr_1, 0) as decimal)/ 12 as mm_insv_yr_1 ,
                    cast(coalesce(insv_yr_2, 0) as decimal)/ 12 as mm_insv_yr_2 ,
                    cast(coalesce(insv_yr_3, 0) as decimal)/ 12 as mm_insv_yr_3 ,
                    0 as mm_insv_yr_dummy ,
                    cast(coalesce(insv_yr_1, 0) as decimal) insv_yr_1_rollup ,
                    cast(coalesce(insv_yr_1, 0) as decimal) + cast(coalesce(insv_yr_2, 0) as decimal) + cast(coalesce(insv_yr_3, 0) as decimal) insv_yr_3_rollup,
                    extract(year from current_date) curr_yr
                from
                    df ) unpivot (mm_insv for mm_type in (mm_insv_yr_1, mm_insv_yr_2, mm_insv_yr_3, mm_insv_yr_dummy))) ),
                         
            gsv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_gsv) over (partition by Project order by year_of_nsv ) next_mm_gsv
            from (
            select flag,upper(Action) as Action,Project,Proj_ID,Launch_Date,Region,gsv_yr_1,gsv_yr_2,gsv_yr_3,
                launch_yr + launch_month_nov_dec_flag launch_year ,
                case
                    when launch_month_nov_dec_flag = 1 then 1
                    else launch_month
                end launch_month ,
                launch_yr + launch_month_nov_dec_flag +
                case
                    when mm_type = 'mm_gsv_yr_1' then 0
                    when mm_type = 'mm_gsv_yr_2' then 1
                    when mm_type = 'mm_gsv_yr_3' then 2
                    when mm_type = 'mm_gsv_yr_dummy' then 3
                    else 0
                end year_of_nsv ,
                mm_gsv *(13-launch_month) yearly_total ,
                mm_gsv *(launch_month-1) mm_gsv ,
                gsv_yr_1_rollup ,
                gsv_yr_3_rollup ,
                curr_yr
            from (             
                select flag,upper(Action) as Action ,Project,Proj_ID,Launch_Date,Region,gsv_yr_1,gsv_yr_2,gsv_yr_3,
                    extract( year from cast(Launch_Date as date)) as launch_yr,
                    extract( month from cast(Launch_Date as date)) as launch_month ,
                    case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                    cast(coalesce(gsv_yr_1, 0) as decimal)/ 12 as mm_gsv_yr_1 ,
                    cast(coalesce(gsv_yr_2, 0) as decimal)/ 12 as mm_gsv_yr_2 ,
                    cast(coalesce(gsv_yr_3, 0) as decimal)/ 12 as mm_gsv_yr_3 ,
                    0 as mm_gsv_yr_dummy ,
                    cast(coalesce(gsv_yr_1, 0) as decimal) gsv_yr_1_rollup ,
                    cast(coalesce(gsv_yr_1, 0) as decimal) + cast(coalesce(gsv_yr_2, 0) as decimal) + cast(coalesce(gsv_yr_3, 0) as decimal) gsv_yr_3_rollup,
                    extract(year from current_date) curr_yr
                from
                    df ) unpivot (mm_gsv for mm_type in (mm_gsv_yr_1, mm_gsv_yr_2, mm_gsv_yr_3, mm_gsv_yr_dummy))) )             

				select insv.flag flag, upper(insv.Action) as Action ,nsv.Project Project,
				nsv.Launch_Date Launch_Date
				,nsv.Region Region
				,nsv.Proj_ID Proj_ID
				,nsv.launch_year launch_yr
				,nsv.launch_month launch_mth
				,nsv.year_of_nsv yr_of_nsv
				,nsv.filtr fltr
				,nsv.nsv_yr_1
				,nsv.nsv_yr_2
				,nsv.nsv_yr_3,
				insv.insv_yr_1,
				insv.insv_yr_2,
				insv.insv_yr_3,
				insv.mm_insv,
				insv.yearly_total as insv_year,
				insv.next_mm_insv as insv_wrap,
                insv.insv_yr_1_rollup,
                insv.insv_yr_3_rollup
				,nsv.yearly_total as nsv_year
				,nsv.next_mm_nsv as nsv_wrap 
				,nsv.nsv_yr_1_rollup
				,nsv.nsv_yr_3_rollup
                , insv.BU, insv.Project_Desc, insv.Portfolio_Bucket, insv.pd_or_ft_days, insv.pkg_days, insv.canblz_pct, insv.gross_mrgn_pct
                ,gsv.yearly_total as gsv_year
                ,gsv.next_mm_gsv as gsv_wrap 
                ,gsv.gsv_yr_1_rollup gsv_yr_1_rollup
                ,gsv.gsv_yr_3_rollup gsv_yr_3_rollup
                ,gsv.gsv_yr_1
                ,gsv.gsv_yr_2
                ,gsv.gsv_yr_3

                from nsv_calc nsv left join insv_calc insv         
                on nsv.Project=insv.Project and nsv.year_of_nsv=insv.year_of_nsv
                left join gsv_calc gsv on nsv.Project=gsv.Project and nsv.year_of_nsv = gsv.year_of_nsv
            )
             """
            ).df()

    #Renaming and dropping few of the columns 
    z_df1.drop(['mm_insv','launch_mth'],axis=1,inplace=True)
    z_df1.rename(columns={'yr_of_nsv':'Year of NSV', 'launch_yr':'Launch Year',
                          'nsv_year':'Simulated NSV Year','insv_year':'Simulated iNSV Year','fltr':'Filter','nsv_wrap':'Simulated NSV Wrap',
                          'insv_wrap':'Simulated iNSV Wrap','insv_yr_1': 'Simulated iNSV Year 1','insv_yr_2': 'Simulated iNSV Year 2','insv_yr_3': 'Simulated iNSV Year 3','insv_yr_1_rollup': 'Simulated iNSV Year 1 Rollup','insv_yr_3_rollup': 'Simulated iNSV Year 3 Rollup',
                          'nsv_yr_1': 'Simulated NSV Year 1','nsv_yr_2': 'Simulated NSV Year 2','nsv_yr_3': 'Simulated NSV Year 3','nsv_yr_1_rollup': 'Simulated NSV Year 1 Rollup','nsv_yr_3_rollup': 'Simulated NSV Year 3 Rollup','pd_or_ft_days':'Simulated pd or ft Days','pkg_days':'Simulated pkg Days','canblz_pct':'Simulated Canblz Pct',
                          'gross_mrgn_pct':'Simulated Gross Margin Pct',
                          'gsv_yr_1':'Simulated GSV Year 1','gsv_yr_2':'Simulated GSV Year 2','gsv_yr_3':'Simulated GSV Year 3','gsv_yr_3_rollup':'Simulated GSV Year 3 Rollup','gsv_yr_1_rollup':'Simulated GSV Year 1 Rollup','gsv_wrap':'Simulated GSV Wrap','gsv_year':'Simulated GSV Year'}, inplace=True)
    return(z_df1)

def plot_bar(df,measure,dim_selected):
    Published_column='Published '+ measure
    Simulated_column='Simulated '+ measure
    ylabel=measure
    df[Published_column]=df[Published_column].round(2) 
    df[Simulated_column]=df[Simulated_column].round(2)
    grp_by = df.groupby(dim_selected)[[Published_column,Simulated_column]].sum().reset_index()
    if measure == "R&D Days":
        grp_by[Published_column]=grp_by[Published_column]/4
        grp_by[Simulated_column]=grp_by[Simulated_column]/4
    col1,col2=st.columns(2)
    with col1 :
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        # Width of the bars
        bar_width = 0.2
        # Index for the x-axis
        ind = range(len(grp_by))
        # Plotting Published sales
        Published_sales = ax.bar(ind, grp_by[Published_column], bar_width,label='Published')
        Simulated_sales = ax.bar([i + bar_width for i in ind], grp_by[Simulated_column], bar_width, label='Simulated')
        # Setting labels and title
        ax.set_xlabel(dim_selected)
        #ax.set_ylabel(ylabel)
        ax.set_xticks([i + bar_width / 2 for i in ind])
        ax.set_xticklabels(grp_by[dim_selected])
        #ax.set(ylim=(10, 150))
        ax.legend()
        # Show plot
        st.pyplot()
    with col2:
        if measure == "R&D Days":
            grp_by['Published'] = grp_by[Published_column].round(0)
            grp_by['Simulated'] = grp_by[Simulated_column].round(0)
            grp_by['Difference'] = (grp_by[Simulated_column] - grp_by[Published_column]).round(0)
            grp_by['Difference'] = grp_by['Difference'].round(0)
        else :
            grp_by['Published ($)'] = grp_by[Published_column].round(0)
            grp_by['Simulated ($)'] = grp_by[Simulated_column].round(0)
            grp_by['Difference ($)'] = (grp_by[Simulated_column] - grp_by[Published_column]).round(0)
            grp_by['Difference ($)'] = grp_by['Difference ($)'].round(0)
        grp_by['% Difference'] = [0.0 if published == 0 else ((simulated/published)-1)*100 for published, simulated in zip(grp_by[Published_column], grp_by[Simulated_column])]
        grp_by['% Difference']  = grp_by['% Difference'].fillna(0).round(1).astype(str) + '%'
        if dim_selected == 'Portfolio Bucket' :
            grp_by['Published % of Total'] = (grp_by[Published_column]/grp_by[Published_column].sum())*100
            grp_by['Published % of Total'] =grp_by['Published % of Total'].fillna(0)
            grp_by['Published % of Total'] = grp_by['Published % of Total'].round(1).astype(str)+'%'
            grp_by['Simulated % of Total'] = (grp_by[Simulated_column]/grp_by[Simulated_column].sum())*100
            grp_by['Simulated % of Total'] = grp_by['Simulated % of Total'].fillna(0)
            grp_by['Simulated % of Total'] = grp_by['Simulated % of Total'].round(1).astype(str)+'%'
        grp_by = grp_by.drop( columns = [Published_column,Simulated_column], axis =1)
        float_column_names = grp_by.select_dtypes(float).columns
        df_raw[float_column_names] = grp_by[float_column_names].round(0)
        grp_by = grp_by.set_index(dim_selected)
        st.dataframe(grp_by,height=275,width=600)

def plot_gm(df,dim_selected):
    gm = weighted_average_of_group(df,dim_selected)
    ylabel = 'Gross Margin %'  
    gm['Published']=gm['Published Gross Margin WA Pct'].round(2)
    gm['Simulated']=gm['Simulated Gross Margin WA Pct'].round(2)
    col1,col2 = st.columns(2)
    with col1:
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        # Width of the bars
        bar_width = 0.2
        # Index for the x-axis
        ind = range(len(gm))
        # Plotting Published sales
        Published_sales = ax.bar(ind,gm['Published'], bar_width, label='Published')
        Simulated_sales = ax.bar([i + bar_width for i in ind], gm['Simulated'], bar_width, label='Simulated')
        # Setting labels and title
        ax.set_xlabel(dim_selected)
        #ax.set_ylabel(ylabel)
        ax.set_xticks([i + bar_width / 2 for i in ind])
        ax.set_xticklabels(gm[dim_selected])
        #ax.set(ylim=(10, 150))
        ax.legend()
        # Show plot
        st.pyplot()
    with col2:
        gm['Difference'] = gm['Simulated'] - gm['Published']
        gm['Published']=gm['Published'].fillna(0).round(2).astype(str) + '%'
        gm['Simulated']=gm['Simulated'].fillna(0).round(2).astype(str) + '%'
        gm['Difference'] = gm['Difference'].fillna(0).round(0)
        gm=gm.drop(columns = ['Published Gross Margin WA Pct','Simulated Gross Margin WA Pct'] , axis=1)
        float_column_names = gm.select_dtypes(float).columns
        gm[float_column_names] = gm[float_column_names].round(0)
        gm = gm.set_index(dim_selected)
        st.dataframe(gm,height=175,width=600)
    
def validate_main(df):
    # duplicate project names 
    errors = []
    #Validate column names
    expected_columns = ['Action', 'Proj_ID','Project','Project_Desc','Launch_Date','Region','BU','Portfolio_Bucket',
                 'pd_or_ft_days','pkg_days','canblz_pct','gross_mrgn_pct',
                 'nsv_yr_1','nsv_yr_2','nsv_yr_3','insv_yr_1','insv_yr_2','insv_yr_3',
                 'gsv_yr_1','gsv_yr_2','gsv_yr_3']
    if list(df.columns) != expected_columns:
        errors.append("Error: Column names should not change. Please keep the column name as same as the template")
        return errors

    # Validate no duplicate project names
    if df['Project'].duplicated().any():
       errors.append("Error: The Projects you entered already exist. Please change the Simulated Project Name to run the simulator", )

   # Validate no nulls in project names, project ids
    columns_to_check = ['Project', 'Proj_ID','Launch_Date']
    for column in columns_to_check:
        if df[column].isnull().any():
            errors.append(f"Error: There are null values in column '{column}'. Please provide the values.")
        
    #Validate for string values in measure
    if df['nsv_yr_1'].dtype == 'object' or df['nsv_yr_2'].dtype == 'object' or df['nsv_yr_3'].dtype == 'object' :
        errors.append(f"Error: There are non-numerical values in NSV columns. Please provide the numerical values.")
    if df['insv_yr_1'].dtype == 'object' or df['insv_yr_2'].dtype == 'object' or df['insv_yr_3'].dtype == 'object' :
        errors.append(f"Error: There are non-numerical values in iNSV columns. Please provide the numerical values.")
    if df['gsv_yr_1'].dtype == 'object' or df['gsv_yr_2'].dtype == 'object' or df['gsv_yr_3'].dtype == 'object' :
        errors.append(f"Error: There are non-numerical values in GSV columns. Please provide the numerical values.")
    
    return errors

def validate_sub(df):
    errors = []
    df = df[df['Action']=='Add']
    columns_to_check = ['Region','BU','Project_Desc','Portfolio_Bucket']
    for column in columns_to_check:
        if df[column].isnull().any():
            errors.append(f"Warning: There are null values in column '{column}'. Please provide the values for better result.")
    return errors

def plot_comparison(final, measure_selected, group_by):

    st.markdown(f"<span style='font-size:20px;font-family:Source Sans Pro;font-weight:700'>Published vs Simulated by {group_by}</span>",unsafe_allow_html=True)
    if measure_selected == 'Gross Margin %':
        plot_gm(final, group_by)
    else:
        plot_bar(final, measure_selected, group_by)

def delta_cal(num,den):
    if num == 0 and den == 0 :
        delta='0.0%'
    elif den==0 :
        delta='0.0%'
    else :
        try:
            delta1=(num-den).round(0).astype(str)
            delta2 = (((num/den)-1)*100).round(1).astype(str) + '%'
            delta = delta1 + ' ('+ delta2+')'
        except:
            delta1=str(num-den)
            delta2 = str(round((((num/den)-1)*100),1)) + '%'
            delta = delta1 + ' ('+ delta2+')'

    return delta

def kip_cards(df,sim_project_count,publ_project_count):

    NSV =  millify(df['Simulated 1 Year Rolling NSV'].sum(), precision=0)
    iNSV = millify(df['Simulated 1 Year Rolling iNSV'].sum(), precision=0, drop_nulls=False)
    GSV = millify(df['Simulated 1 Year Rolling GSV'].sum(), precision=0, drop_nulls=False)
    NSV_3 =  millify(df['Simulated 3 Year Rolling NSV'].sum(), precision=0)
    iNSV_3 = millify(df['Simulated 3 Year Rolling iNSV'].sum(), precision=0, drop_nulls=False)
    GSV_3 = millify(df['Simulated 3 Year Rolling GSV'].sum(), precision=0, drop_nulls=False)


    col1,col2,col3=st.columns(3)
    with col1:
        st.metric(label='Simulated 1 Year Rolling NSV ($)', value=NSV, delta=delta_cal(df['Simulated 1 Year Rolling NSV'].sum(),df['Published 1 Year Rolling NSV'].sum()))
    with col2:
        st.metric(label='Simulated 1 Year Rolling iNSV ($)', value=iNSV, delta=delta_cal(df['Simulated 1 Year Rolling iNSV'].sum() , df['Published 1 Year Rolling iNSV'].sum()))
    with col3:
        st.metric(label='Simulated 1 Year Rolling GSV ($)', value=GSV, delta= delta_cal(df['Simulated 1 Year Rolling GSV'].sum(),df['Published 1 Year Rolling GSV'].sum()))
    
    col1,col2,col3 = st.columns(3)
    with col2:
        st.metric(label='Simulated 3 Year Rolling iNSV ($)', value=iNSV_3, delta = delta_cal(df['Simulated 3 Year Rolling iNSV'].sum(),df['Published 3 Year Rolling iNSV'].sum()))
    with col3:
        st.metric(label='Simulated 3 Year Rolling GSV ($)', value=GSV_3, delta=delta_cal(df['Simulated 3 Year Rolling GSV'].sum(), df['Published 3 Year Rolling GSV'].sum()))
    with col1:
        st.metric(label='Simulated 3 Year Rolling NSV ($)', value=NSV_3, delta = delta_cal(df['Simulated 3 Year Rolling NSV'].sum(),df['Published 3 Year Rolling NSV'].sum()))
    
    col1,col2,col3=st.columns(3)
    with col1:
        st.metric(label='Simulated R&D Days', value=(df['Simulated R&D Days'].sum()/4).astype('int64'), delta=delta_cal((df['Simulated R&D Days'].sum()/4),(df['Published R&D Days'].sum()/4)))
    with col2:
        st.metric(label='Simulated Project Count',value = sim_project_count.nunique(), delta =delta_cal(sim_project_count.nunique(),publ_project_count.nunique()))
    with col3:
        df1 = df[df['Filter']==1]
        df1['Weighted_Score1'] = df1['Published Gross Margin Pct'] * df1['Published NSV Year 1']
        df1['Weighted_Score2'] = df1['Simulated Gross Margin Pct'] * df1['Simulated NSV Year 1']
        if  df1['Weighted_Score1'].sum()==0  and df1['Published NSV Year 1'].sum() ==0: 0
        else: wa_Published =  (df1['Weighted_Score1'].sum()/df1['Published NSV Year 1'].sum()).round(1)
        if df1['Weighted_Score2'].sum() == 0 and df1['Simulated NSV Year 1'].sum() ==0: 0
        else: wa_Simulated = (df1['Weighted_Score2'].sum()/df1['Simulated NSV Year 1'].sum()).round(1)
        st.metric(label='Simulated Gross Margin %', value=wa_Simulated.astype(str) + '%', delta=(wa_Simulated - wa_Published).round(2),help="Difference in pts")

def main():
    #st.title('Kellogg POC Simulator')
    st.markdown("<span style='color:#f60b45;font-size:44px;font-family:Source Sans Pro;font-weight:700'>Pre-launch Data Simulator</span>",
             unsafe_allow_html=True)

    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Kellanova_logo.svg/1200px-Kellanova_logo.svg.png",width=150)
    
    st.sidebar.write('The Simulator helps user to understand the different scenario for Pre-Launched Data.\n\n Users can Add, Edit and/or Delete any number of projects to compare')
    # Download Functionality 
    st.subheader('1. Upload Simulation File')
    st.write('Download the  latest Pre-Launch data and add projects for simulation.')
    st.download_button("Download the Pre Launch Data",df_raw.to_csv(index=True),file_name="Pre_Launch Data.csv",mime="text/csv")

    #upload user input file
    st.write('Upload  the updated pre-launch data file with simulation projects.')
    uploaded_file_df1 = st.file_uploader(" \* Only CSV files are supported", type=["csv"])

    if uploaded_file_df1:
        df1 = pd.read_csv(uploaded_file_df1)

        # filtering the ADD rows only
        df1 = df1[df1['Action'].str.upper()=='ADD']
        df_concat = pd.concat([df1,df_raw_copy])

        if len(validate_main(df_concat))>0:
            for i in validate_main(df_concat):
                st.error(i)
        else :
            # validation
            for i in validate_sub(df_concat):
                st.warning(i)
            # changing the data type and format of launch date
            df_concat['Launch_Date'] = pd.to_datetime(df_concat['Launch_Date'])
            df_concat['Launch_Year'] = df_concat['Launch_Date'].apply(lambda x: x.year if x.month < 11 else x.year + 1)
            df_concat['Launch_Date'] = df_concat['Launch_Date'].dt.strftime('%Y-%m-%d')
            df_concat['Launch_Year'] = df_concat['Launch_Year'].astype('Int64').astype(str)
            df_concat['Proj_ID'] = df_concat['Proj_ID'].astype(str)

            user_input_data = copy.deepcopy(df_concat)
            st.subheader('2. Start Modelling Simulator')
            st.write('Check the data before starting the Simulation. You can include/exclude projects for desired results.')

            # Checkbox 
            df_concat=df_concat.sort_values(by ='Action',ascending=True)
            df_concat['flag']=True
            df_concat.insert(0, 'flag', df_concat.pop('flag'))
            df_concat.insert(1, 'Project', df_concat.pop('Project')) 
            q=st.data_editor(df_concat,    column_config={
            "flag": st.column_config.CheckboxColumn(
                "Include",
                help="Select the projects you want to delete",
                default=False,
            )},disabled=['Project','Proj_ID','Action','Launch_Year','Launch_Date','Region','nsv_yr_1','nsv_yr_2','nsv_yr_3'],hide_index=True,)
            #df_deletion_Simulated=q[q['flag']==False]
            #st.write('q',q)

            # Button to simulate
            button_pressed = st.button("Start the Simulation",help="Simulation takes few seconds to process the data")

            if button_pressed or st.session_state.load_state:
                st.session_state.load_state=True
                q=q.sort_values(by ='Launch_Year',ascending=True)
                st.sidebar.subheader('Filter Pane',help='This Pane is only applicable for the section 3.Simulation Result')
                measure_selected = st.sidebar.selectbox("Select the Measure: ",
                                                    ('1 Year Rolling NSV','1 Year Rolling iNSV','1 Year Rolling GSV','3 Year Rolling NSV','3 Year Rolling iNSV','3 Year Rolling GSV',
                                                    'R&D Days','Gross Margin %'),help="This selector is not applicable on Scorecard Section")
                dynamic_filters = DynamicFilters(q, filters=['Launch_Year', 'Region','Portfolio_Bucket','BU','Project'])
                dynamic_filters.display_filters(location='sidebar')
                df_filtered = dynamic_filters.filter_df()

                # Processing the data 
                sql_result = sql_process(df_filtered)
                final,sim_project_count,publ_project_count = Process_data(sql_result,raw_copy)
                final_copy = copy.deepcopy(final)
                final.sort_values(by='Project ID',ascending=False,inplace=True)
                final.drop(['flag','gm_pct_yr_1','gm_pct_yr_2','gm_pct_yr_3','file_nm','kortex_upld_ts'],axis=1,inplace=True)

                st.subheader('3. Simulation Result')

                # KPI Cards
                st.info('Use the filter pane to dynamically adjust the simulation results.', icon="ℹ️")
                st.markdown("<span style='font-size:25px;font-family:Source Sans Pro;font-weight:700'>Scorecard</span>",unsafe_allow_html=True,help="Comparison with the Published Value")
                kip_cards(final,sim_project_count,publ_project_count)
                st.markdown("-----")
                # Graphs
                st.markdown(f"<span style='font-size:25px;font-family:Source Sans Pro;font-weight:700'>Detail Analysis </span> ",unsafe_allow_html=True,help="Use the Filter pane to change the Measure & filter the values")
                st.markdown(f"<span style='font-size:20px;font-family:Source Sans Pro;font-weight:400'>Measure Selected - </span> <span style='font-size:20px;font-family:Source Sans Pro;color:#f60b45;font-weight:400'> {measure_selected}</span>",unsafe_allow_html=True)
                # st.info('Change the selected measure from the filter pane.', icon="ℹ️")
                #Graph sections
                plot_comparison(final, measure_selected, 'Region')
                if measure_selected not in ['R&D Days','Gross Margin %']:
                    plot_comparison(final, measure_selected, 'Year of NSV')
                plot_comparison(final, measure_selected, 'BU')
                plot_comparison(final, measure_selected, 'Launch Year')
                plot_comparison(final, measure_selected, 'Portfolio Bucket')
                st.subheader('The Simulated Data') 
                final.set_index('Action', inplace=True)
                st.write(final)
                st.download_button("Download Simulated Data",final.to_csv(index=False),file_name="Simulator Output.csv",mime="text/csv")

if __name__ == "__main__":
    main()
