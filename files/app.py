import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Teacher Scheduling Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

EXCEL_FILE_PATH = 'teacher_scheduling_template.xlsx'

# ==================== AUTHENTICATION ====================
# Load users from secrets
USERS = st.secrets["passwords"]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password():
    def password_entered():
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        if username in USERS and USERS[username] == hash_password(password):
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = username
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🔐 Teacher Scheduling Login")
        st.markdown("---")
        
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        
        if st.button("Login", type="primary", width="stretch"):
            password_entered()
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Username or password incorrect")
        
        st.markdown("---")

    return False

if not check_password():
    st.stop()

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'optimization_run' not in st.session_state:
    st.session_state.optimization_run = False
if 'schedule_df' not in st.session_state:
    st.session_state.schedule_df = None
if 'workload_df' not in st.session_state:
    st.session_state.workload_df = None
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = None
if 'data_modified' not in st.session_state:
    st.session_state.data_modified = False

# ==================== FUNCTIONS ====================
def load_data():
    try:
        teachers = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Teachers', engine='openpyxl')
        availability = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Teacher_Availability', engine='openpyxl')
        skills = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Teacher_Skills', engine='openpyxl')
        requirements = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Class_Requirements', engine='openpyxl')
        return teachers, availability, skills, requirements, None
    except Exception as e:
        return None, None, None, None, str(e)

def save_data(teachers, availability, skills, requirements):
    try:
        with pd.ExcelWriter(EXCEL_FILE_PATH, engine='openpyxl') as writer:
            teachers.to_excel(writer, sheet_name='Teachers', index=False)
            availability.to_excel(writer, sheet_name='Teacher_Availability', index=False)
            skills.to_excel(writer, sheet_name='Teacher_Skills', index=False)
            requirements.to_excel(writer, sheet_name='Class_Requirements', index=False)
        return True, None
    except Exception as e:
        return False, str(e)

def run_optimization(teachers, availability, skills, requirements):
    availability_full = availability.merge(teachers[['Teacher_Name', 'Teacher_Type']], on='Teacher_Name')
    skills_full = skills.merge(teachers[['Teacher_Name', 'Teacher_Type']], on='Teacher_Name')
    
    requirements['Class_ID'] = requirements.apply(lambda x: f"{x['Class_Type']}_{x['Day']}_{x['Time_Slot']}", axis=1)
    requirements['TimeSlot_ID'] = requirements.apply(lambda x: f"{x['Day']}_{x['Time_Slot']}", axis=1)
    availability_full['TimeSlot_ID'] = availability_full.apply(lambda x: f"{x['Day']}_{x['Time_Slot']}", axis=1)
    
    model = LpProblem("Teacher_Scheduling", LpMaximize)
    teacher_names = teachers['Teacher_Name'].tolist()
    class_ids = requirements['Class_ID'].tolist()
    
    x = LpVariable.dicts("assign", [(t, c) for t in teacher_names for c in class_ids], cat='Binary')
    class_fully_staffed = LpVariable.dicts("fully_staffed", class_ids, cat='Binary')
    class_lead_shortage = LpVariable.dicts("lead_shortage", class_ids, lowBound=0, cat='Integer')
    class_assistant_shortage = LpVariable.dicts("assistant_shortage", class_ids, lowBound=0, cat='Integer')
    
    model += (100 * lpSum([class_fully_staffed[c] for c in class_ids]) - 
              50 * lpSum([class_lead_shortage[c] for c in class_ids]) -
              30 * lpSum([class_assistant_shortage[c] for c in class_ids]), "Maximize_Coverage")
    
    for idx, req in requirements.iterrows():
        class_id = req['Class_ID']
        time_slot = req['TimeSlot_ID']
        class_type = req['Class_Type']
        
        eligible_leads = []
        eligible_assistants = []
        
        for teacher_name in teacher_names:
            is_available = availability_full[(availability_full['Teacher_Name'] == teacher_name) &
                                            (availability_full['TimeSlot_ID'] == time_slot) &
                                            (availability_full['Available'] == 1)].shape[0] > 0
            can_teach = skills_full[(skills_full['Teacher_Name'] == teacher_name) &
                                   (skills_full['Class_Type'] == class_type) &
                                   (skills_full['Can_Teach'] == 1)].shape[0] > 0
            
            if is_available and can_teach:
                teacher_type = teachers[teachers['Teacher_Name'] == teacher_name]['Teacher_Type'].values[0]
                if teacher_type == 'Lead':
                    eligible_leads.append(teacher_name)
                else:
                    eligible_assistants.append(teacher_name)
        
        if eligible_leads:
            model += lpSum([x[t, class_id] for t in eligible_leads]) + class_lead_shortage[class_id] >= req['Lead_Teachers_Required']
        else:
            model += class_lead_shortage[class_id] >= req['Lead_Teachers_Required']
        
        if eligible_assistants:
            model += lpSum([x[t, class_id] for t in eligible_assistants]) + class_assistant_shortage[class_id] >= req['Assistant_Teachers_Required']
        else:
            model += class_assistant_shortage[class_id] >= req['Assistant_Teachers_Required']
        
        if eligible_leads and eligible_assistants:
            total_required = req['Lead_Teachers_Required'] + req['Assistant_Teachers_Required']
            total_assigned = lpSum([x[t, class_id] for t in (eligible_leads + eligible_assistants)])
            model += total_assigned >= total_required * class_fully_staffed[class_id]
            model += class_fully_staffed[class_id] <= total_assigned / total_required
    
    for teacher_name in teacher_names:
        for time_slot in requirements['TimeSlot_ID'].unique():
            classes_at_time = requirements[requirements['TimeSlot_ID'] == time_slot]['Class_ID'].tolist()
            model += lpSum([x[teacher_name, c] for c in classes_at_time]) <= 1
    
    for teacher_name in teacher_names:
        for idx, req in requirements.iterrows():
            class_id = req['Class_ID']
            time_slot = req['TimeSlot_ID']
            class_type = req['Class_Type']
            
            is_available = availability_full[(availability_full['Teacher_Name'] == teacher_name) &
                                            (availability_full['TimeSlot_ID'] == time_slot) &
                                            (availability_full['Available'] == 1)].shape[0] > 0
            can_teach = skills_full[(skills_full['Teacher_Name'] == teacher_name) &
                                   (skills_full['Class_Type'] == class_type) &
                                   (skills_full['Can_Teach'] == 1)].shape[0] > 0
            
            if not (is_available and can_teach):
                model += x[teacher_name, class_id] == 0
    
    for teacher_name in teacher_names:
        max_hours = teachers[teachers['Teacher_Name'] == teacher_name]['Max_Hours_Per_Week'].values[0]
        model += lpSum([x[teacher_name, c] for c in class_ids]) <= max_hours
    
    model.solve(PULP_CBC_CMD(msg=0))
    
    if model.status != 1:
        return None, None, "Could not find optimal solution"
    
    schedule_data = []
    teacher_assignments = {}
    
    for idx, req in requirements.iterrows():
        class_id = req['Class_ID']
        assigned_leads = []
        assigned_assistants = []
        
        for teacher_name in teacher_names:
            if value(x[teacher_name, class_id]) == 1:
                teacher_type = teachers[teachers['Teacher_Name'] == teacher_name]['Teacher_Type'].values[0]
                if teacher_type == 'Lead':
                    assigned_leads.append(teacher_name)
                else:
                    assigned_assistants.append(teacher_name)
                
                if teacher_name not in teacher_assignments:
                    teacher_assignments[teacher_name] = []
                teacher_assignments[teacher_name].append(class_id)
        
        lead_shortage = int(value(class_lead_shortage[class_id]))
        assistant_shortage = int(value(class_assistant_shortage[class_id]))
        
        if lead_shortage == 0 and assistant_shortage == 0:
            status = 'Fully Staffed'
        elif lead_shortage > 0:
            status = f'Missing {lead_shortage} Lead'
        else:
            status = f'Missing {assistant_shortage} Assistant(s)'
        
        schedule_data.append({
            'Day': req['Day'], 'Time_Slot': req['Time_Slot'], 'Class_Type': req['Class_Type'],
            'Children': req['Number_of_Children'], 'Lead_Required': req['Lead_Teachers_Required'],
            'Lead_Assigned': ', '.join(assigned_leads) if assigned_leads else 'NONE',
            'Assistant_Required': req['Assistant_Teachers_Required'],
            'Assistant_Assigned': ', '.join(assigned_assistants) if assigned_assistants else 'NONE',
            'Status': status
        })
    
    schedule_df = pd.DataFrame(schedule_data)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    schedule_df['Day'] = pd.Categorical(schedule_df['Day'], categories=day_order, ordered=True)
    schedule_df = schedule_df.sort_values(['Day', 'Time_Slot'])
    
    workload_data = []
    for teacher_name in teacher_names:
        classes_assigned = len(teacher_assignments.get(teacher_name, []))
        max_hours = teachers[teachers['Teacher_Name'] == teacher_name]['Max_Hours_Per_Week'].values[0]
        teacher_type = teachers[teachers['Teacher_Name'] == teacher_name]['Teacher_Type'].values[0]
        utilization = (classes_assigned / max_hours) * 100 if max_hours > 0 else 0
        workload_data.append({
            'Teacher_Name': teacher_name, 'Teacher_Type': teacher_type,
            'Classes_Assigned': classes_assigned, 'Max_Hours': max_hours,
            'Utilization_%': round(utilization, 1)
        })
    
    workload_df = pd.DataFrame(workload_data).sort_values('Classes_Assigned', ascending=False)
    return schedule_df, workload_df, None

# ==================== MAIN APP ====================
st.markdown('<h1 class="main-header">📚 Teacher Scheduling Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.success(f"👤 Logged in as: **{st.session_state['current_user']}**")
    
    if st.button("🚪 Logout", width="stretch"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    st.info(f"📁 Using file: `{EXCEL_FILE_PATH}`")
    st.divider()
    
    if st.button("🚀 Run Optimization", type="primary", width="stretch"):
        with st.spinner("Running optimization..."):
            teachers, availability, skills, requirements, error = load_data()
            if error:
                st.error(f"Error loading data: {error}")
            else:
                schedule_df, workload_df, opt_error = run_optimization(teachers, availability, skills, requirements)
                if opt_error:
                    st.error(opt_error)
                else:
                    st.session_state.schedule_df = schedule_df
                    st.session_state.workload_df = workload_df
                    st.session_state.optimization_run = True
                    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.data_modified = False
                    st.success("✓ Optimization complete!")
                    st.rerun()
    
    if st.session_state.last_updated:
        st.caption(f"Last updated: {st.session_state.last_updated}")
    if st.session_state.data_modified:
        st.warning("⚠️ Data modified. Re-run optimization.")

# Main content
tab_edit, tab_schedule, tab_stats, tab_detailed, tab_workload, tab_issues = st.tabs([
    "✏️ Edit Data", "📅 Weekly Schedule", "📊 Daily Stats", "📋 Detailed Schedule", "👥 Teacher Workload", "⚠️ Issues"
])

with tab_edit:
    st.header("✏️ Edit Schedule Data")
    st.info("💡 Use filters to find records, make changes, then click 'Save Changes' and 'Run Optimization'.")
    
    teachers, availability, skills, requirements, error = load_data()
    
    if error:
        st.error(f"Error loading data: {error}")
    else:
        # Teachers Section
        st.subheader("👨‍🏫 Teachers")
        with st.expander("🔍 Filter Teachers", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                filter_teacher_type = st.multiselect("Filter by Type", options=teachers['Teacher_Type'].unique().tolist(),
                                                     default=teachers['Teacher_Type'].unique().tolist(), key="filter_teacher_type")
            with col2:
                filter_teacher_search = st.text_input("Search by Name", "", key="filter_teacher_search")
        
        filtered_teachers = teachers[teachers['Teacher_Type'].isin(filter_teacher_type)]
        if filter_teacher_search:
            filtered_teachers = filtered_teachers[filtered_teachers['Teacher_Name'].str.contains(filter_teacher_search, case=False, na=False)]
        
        st.caption(f"Showing {len(filtered_teachers)} of {len(teachers)} teachers")
        edited_teachers = st.data_editor(filtered_teachers, width="stretch", num_rows="dynamic",
                                        column_config={
                                            "Teacher_Name": st.column_config.TextColumn("Teacher Name", required=True),
                                            "Teacher_Type": st.column_config.SelectboxColumn("Type", options=["Lead", "Assistant"], required=True),
                                            "Max_Hours_Per_Week": st.column_config.NumberColumn("Max Hours/Week", min_value=1, max_value=80, required=True)
                                        }, key="teachers_editor")
        
        st.divider()
        
        # Availability Section
        st.subheader("📅 Teacher Availability")
        with st.expander("🔍 Filter Availability", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_avail_teachers = st.multiselect("Filter by Teacher", options=sorted(availability['Teacher_Name'].unique().tolist()),
                                                      default=availability['Teacher_Name'].unique().tolist(), key="filter_avail_teachers")
            with col2:
                filter_avail_days = st.multiselect("Filter by Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                                   default=availability['Day'].unique().tolist(), key="filter_avail_days")
            with col3:
                filter_avail_slots = st.multiselect("Filter by Time Slot", options=sorted(availability['Time_Slot'].unique().tolist()),
                                                    default=availability['Time_Slot'].unique().tolist(), key="filter_avail_slots")
        
        filtered_availability = availability[(availability['Teacher_Name'].isin(filter_avail_teachers)) &
                                            (availability['Day'].isin(filter_avail_days)) &
                                            (availability['Time_Slot'].isin(filter_avail_slots))]
        
        st.caption(f"Showing {len(filtered_availability)} of {len(availability)} records")
        edited_availability = st.data_editor(filtered_availability, width="stretch", num_rows="dynamic",
                                           column_config={
                                               "Teacher_Name": st.column_config.TextColumn("Teacher Name", required=True),
                                               "Day": st.column_config.SelectboxColumn("Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], required=True),
                                               "Time_Slot": st.column_config.TextColumn("Time Slot", required=True),
                                               "Available": st.column_config.NumberColumn("Available (1=Yes, 0=No)", min_value=0, max_value=1, required=True)
                                           }, key="availability_editor")
        
        st.divider()
        
        # Skills Section
        st.subheader("🎓 Teacher Skills")
        with st.expander("🔍 Filter Skills", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                filter_skills_teachers = st.multiselect("Filter by Teacher", options=sorted(skills['Teacher_Name'].unique().tolist()),
                                                       default=skills['Teacher_Name'].unique().tolist(), key="filter_skills_teachers")
            with col2:
                filter_skills_classes = st.multiselect("Filter by Class Type", options=sorted(skills['Class_Type'].unique().tolist()),
                                                      default=skills['Class_Type'].unique().tolist(), key="filter_skills_classes")
        
        filtered_skills = skills[(skills['Teacher_Name'].isin(filter_skills_teachers)) & (skills['Class_Type'].isin(filter_skills_classes))]
        st.caption(f"Showing {len(filtered_skills)} of {len(skills)} records")
        edited_skills = st.data_editor(filtered_skills, width="stretch", num_rows="dynamic",
                                      column_config={
                                          "Teacher_Name": st.column_config.TextColumn("Teacher Name", required=True),
                                          "Class_Type": st.column_config.TextColumn("Class Type", required=True),
                                          "Can_Teach": st.column_config.NumberColumn("Can Teach (1=Yes, 0=No)", min_value=0, max_value=1, required=True)
                                      }, key="skills_editor")
        
        st.divider()
        
        # Requirements Section
        st.subheader("📚 Class Requirements")
        with st.expander("🔍 Filter Class Requirements", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_req_classes = st.multiselect("Filter by Class Type", options=sorted(requirements['Class_Type'].unique().tolist()),
                                                   default=requirements['Class_Type'].unique().tolist(), key="filter_req_classes")
            with col2:
                filter_req_days = st.multiselect("Filter by Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                                default=requirements['Day'].unique().tolist(), key="filter_req_days")
            with col3:
                filter_req_slots = st.multiselect("Filter by Time Slot", options=sorted(requirements['Time_Slot'].unique().tolist()),
                                                 default=requirements['Time_Slot'].unique().tolist(), key="filter_req_slots")
        
        filtered_requirements = requirements[(requirements['Class_Type'].isin(filter_req_classes)) &
                                            (requirements['Day'].isin(filter_req_days)) &
                                            (requirements['Time_Slot'].isin(filter_req_slots))]
        
        st.caption(f"Showing {len(filtered_requirements)} of {len(requirements)} records")
        edited_requirements = st.data_editor(filtered_requirements, width="stretch", num_rows="dynamic",
                                           column_config={
                                               "Class_Type": st.column_config.TextColumn("Class Type", required=True),
                                               "Day": st.column_config.SelectboxColumn("Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], required=True),
                                               "Time_Slot": st.column_config.TextColumn("Time Slot", required=True),
                                               "Number_of_Children": st.column_config.NumberColumn("# Children", min_value=1, required=True),
                                               "Teacher_Child_Ratio": st.column_config.TextColumn("Ratio", required=True),
                                               "Lead_Teachers_Required": st.column_config.NumberColumn("Lead Teachers", min_value=0, required=True),
                                               "Assistant_Teachers_Required": st.column_config.NumberColumn("Assistant Teachers", min_value=0, required=True)
                                           }, key="requirements_editor")
        
        st.divider()
        
        # Save buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Changes", type="primary", width="stretch"):
                teachers_to_save = teachers[~teachers['Teacher_Type'].isin(filter_teacher_type)]
                if filter_teacher_search:
                    teachers_to_save = pd.concat([teachers_to_save, teachers[~teachers['Teacher_Name'].str.contains(filter_teacher_search, case=False, na=False)]]).drop_duplicates()
                teachers_to_save = pd.concat([teachers_to_save, edited_teachers]).drop_duplicates(subset=['Teacher_Name'], keep='last')
                
                availability_to_save = availability[~(availability['Teacher_Name'].isin(filter_avail_teachers)) |
                                                   ~(availability['Day'].isin(filter_avail_days)) |
                                                   ~(availability['Time_Slot'].isin(filter_avail_slots))]
                availability_to_save = pd.concat([availability_to_save, edited_availability]).drop_duplicates()
                
                skills_to_save = skills[~(skills['Teacher_Name'].isin(filter_skills_teachers)) | ~(skills['Class_Type'].isin(filter_skills_classes))]
                skills_to_save = pd.concat([skills_to_save, edited_skills]).drop_duplicates()
                
                requirements_to_save = requirements[~(requirements['Class_Type'].isin(filter_req_classes)) |
                                                   ~(requirements['Day'].isin(filter_req_days)) |
                                                   ~(requirements['Time_Slot'].isin(filter_req_slots))]
                requirements_to_save = pd.concat([requirements_to_save, edited_requirements]).drop_duplicates()
                
                success, save_error = save_data(teachers_to_save, availability_to_save, skills_to_save, requirements_to_save)
                if success:
                    st.session_state.data_modified = True
                    st.success("✅ Changes saved! Click 'Run Optimization' in sidebar.")
                else:
                    st.error(f"Error: {save_error}")
        
        with col2:
            if st.button("🔄 Reload Data", width="stretch"):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Filters", width="stretch"):
                for key in st.session_state.keys():
                    if key.startswith('filter_'):
                        del st.session_state[key]
                st.rerun()

# Show message if optimization not run yet
if not st.session_state.optimization_run:
    with tab_schedule:
        st.info("👆 Click 'Run Optimization' in the sidebar to generate the schedule")
    with tab_stats:
        st.info("👆 Click 'Run Optimization' in the sidebar to generate statistics")
    with tab_detailed:
        st.info("👆 Click 'Run Optimization' in the sidebar to view detailed schedule")
    with tab_workload:
        st.info("👆 Click 'Run Optimization' in the sidebar to view teacher workload")
    with tab_issues:
        st.info("👆 Click 'Run Optimization' in the sidebar to identify issues")
else:
    schedule_df = st.session_state.schedule_df
    workload_df = st.session_state.workload_df
    
    # Summary metrics (shown in all tabs)
    total_classes = len(schedule_df)
    fully_staffed = len(schedule_df[schedule_df['Status'] == 'Fully Staffed'])
    understaffed = total_classes - fully_staffed
    
    with tab_schedule:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Classes", total_classes)
        with col2:
            st.metric("✅ Fully Staffed", fully_staffed, delta=f"{(fully_staffed/total_classes)*100:.0f}%")
        with col3:
            st.metric("⚠️ Understaffed", understaffed)
        with col4:
            st.metric("Avg Utilization", f"{workload_df['Utilization_%'].mean():.1f}%")
        
        st.divider()
        st.subheader("Weekly Calendar View")
        
        days = schedule_df['Day'].cat.categories.tolist()
        time_slots = sorted(schedule_df['Time_Slot'].unique())
        
        for time_slot in time_slots:
            st.markdown(f"### ⏰ {time_slot}")
            cols = st.columns(len(days))
            
            for i, day in enumerate(days):
                with cols[i]:
                    st.markdown(f"**{day}**")
                    classes_at_slot = schedule_df[(schedule_df['Day'] == day) & (schedule_df['Time_Slot'] == time_slot)]
                    
                    if not classes_at_slot.empty:
                        for _, cls in classes_at_slot.iterrows():
                            box_class = 'success-box' if cls['Status'] == 'Fully Staffed' else 'warning-box'
                            icon = '✅' if cls['Status'] == 'Fully Staffed' else '⚠️'
                            age_icon = {'Toddler': '👶', 'Pre-K': '🧒', 'Elementary': '👦', 'Infant': '🍼'}.get(cls['Class_Type'], '📚')
                            
                            st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
                            st.markdown(f"**{icon} {age_icon} {cls['Class_Type']}**")
                            st.caption(f"👥 {cls['Children']} children")
                            if cls['Lead_Assigned'] != 'NONE':
                                st.caption(f"👨‍🏫 {cls['Lead_Assigned']}")
                            if cls['Assistant_Assigned'] != 'NONE':
                                st.caption(f"👤 {cls['Assistant_Assigned']}")
                            if cls['Status'] != 'Fully Staffed':
                                st.caption(f"⚠️ {cls['Status']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No classes*")
            st.divider()
    
    with tab_stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Classes", total_classes)
        with col2:
            st.metric("✅ Fully Staffed", fully_staffed)
        with col3:
            st.metric("⚠️ Understaffed", understaffed)
        with col4:
            st.metric("Avg Utilization", f"{workload_df['Utilization_%'].mean():.1f}%")
        
        st.divider()
        st.subheader("📊 Daily Statistics")
        
        days = schedule_df['Day'].cat.categories.tolist()
        
        for day in days:
            st.markdown(f"### 📅 {day}")
            day_classes = schedule_df[schedule_df['Day'] == day]
            
            if day_classes.empty:
                st.info(f"No classes on {day}")
                continue
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Classes", len(day_classes))
            with col2:
                st.metric("Total Children", day_classes['Children'].sum())
            with col3:
                staffed = len(day_classes[day_classes['Status'] == 'Fully Staffed'])
                st.metric("✅ Fully Staffed", staffed)
            with col4:
                st.metric("⚠️ Understaffed", len(day_classes) - staffed)
            
            with st.expander(f"View {day} Details"):
                st.dataframe(day_classes[['Time_Slot', 'Class_Type', 'Children', 'Lead_Assigned', 'Assistant_Assigned', 'Status']], 
                           width="stretch", hide_index=True)
            
            st.divider()
        
        # Weekly charts
        st.markdown("### 📈 Weekly Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            daily_children = schedule_df.groupby('Day')['Children'].sum()
            fig1 = go.Figure(data=[go.Bar(x=daily_children.index.tolist(), y=daily_children.values, marker_color='#1f77b4')])
            fig1.update_layout(title='Total Children per Day', xaxis_title='Day', yaxis_title='Children', height=300)
            st.plotly_chart(fig1, width="stretch")
        
        with col2:
            daily_classes = schedule_df.groupby('Day').size()
            fig2 = go.Figure(data=[go.Bar(x=daily_classes.index.tolist(), y=daily_classes.values, marker_color='#ff7f0e')])
            fig2.update_layout(title='Total Classes per Day', xaxis_title='Day', yaxis_title='Classes', height=300)
            st.plotly_chart(fig2, width="stretch")
    
    with tab_detailed:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Classes", total_classes)
        with col2:
            st.metric("✅ Fully Staffed", fully_staffed)
        with col3:
            st.metric("⚠️ Understaffed", understaffed)
        with col4:
            st.metric("Avg Utilization", f"{workload_df['Utilization_%'].mean():.1f}%")
        
        st.divider()
        st.subheader("Detailed Schedule")
        
        days = schedule_df['Day'].cat.categories.tolist()
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_days = st.multiselect("Filter by Day", days, default=days)
        with col2:
            age_groups = schedule_df['Class_Type'].unique().tolist()
            selected_age_groups = st.multiselect("Filter by Age Group", age_groups, default=age_groups)
        with col3:
            selected_status = st.multiselect("Filter by Status", schedule_df['Status'].unique(), default=schedule_df['Status'].unique())
        
        filtered_schedule = schedule_df[(schedule_df['Day'].isin(selected_days)) & 
                                       (schedule_df['Class_Type'].isin(selected_age_groups)) &
                                       (schedule_df['Status'].isin(selected_status))]
        
        st.dataframe(filtered_schedule, width="stretch", hide_index=True)
        
        csv = filtered_schedule.to_csv(index=False)
        st.download_button("📥 Download Schedule as CSV", data=csv, file_name="teacher_schedule.csv", mime="text/csv")
    
    with tab_workload:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Classes", total_classes)
        with col2:
            st.metric("✅ Fully Staffed", fully_staffed)
        with col3:
            st.metric("⚠️ Understaffed", understaffed)
        with col4:
            st.metric("Avg Utilization", f"{workload_df['Utilization_%'].mean():.1f}%")
        
        st.divider()
        st.subheader("Teacher Workload Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(workload_df, x='Teacher_Name', y='Utilization_%', color='Teacher_Type',
                        title='Teacher Utilization', labels={'Utilization_%': 'Utilization (%)', 'Teacher_Name': 'Teacher'},
                        color_discrete_map={'Lead': '#1f77b4', 'Assistant': '#ff7f0e'})
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Max Capacity")
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("### Summary Stats")
            st.metric("Most Utilized", workload_df.iloc[0]['Teacher_Name'], f"{workload_df.iloc[0]['Utilization_%']}%")
            st.metric("Least Utilized", workload_df.iloc[-1]['Teacher_Name'], f"{workload_df.iloc[-1]['Utilization_%']}%")
            underutilized = workload_df[workload_df['Utilization_%'] < 50]
            st.metric("Underutilized (<50%)", len(underutilized))
        
        st.divider()
        st.dataframe(workload_df, width="stretch", hide_index=True)
    
    with tab_issues:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Classes", total_classes)
        with col2:
            st.metric("✅ Fully Staffed", fully_staffed)
        with col3:
            st.metric("⚠️ Understaffed", understaffed)
        with col4:
            st.metric("Avg Utilization", f"{workload_df['Utilization_%'].mean():.1f}%")
        
        st.divider()
        st.subheader("Issues & Recommendations")
        
        understaffed_df = schedule_df[schedule_df['Status'] != 'Fully Staffed']
        
        if understaffed_df.empty:
            st.success("🎉 Perfect! All classes are fully staffed!")
        else:
            st.warning(f"⚠️ {len(understaffed_df)} classes need attention")
            
            st.markdown("### Understaffed Classes")
            st.dataframe(understaffed_df, width="stretch", hide_index=True)
            
            st.markdown("### 💡 Recommendations")
            
            missing_leads = understaffed_df[understaffed_df['Status'].str.contains('Lead', na=False)]
            missing_assistants = understaffed_df[understaffed_df['Status'].str.contains('Assistant', na=False)]
            
            if not missing_leads.empty:
                st.markdown(f"🔴 **{len(missing_leads)} classes missing Lead Teachers**")
                st.info("Consider hiring more lead teachers or adjusting lead teacher availability")
            
            if not missing_assistants.empty:
                st.markdown(f"🟡 **{len(missing_assistants)} classes missing Assistant Teachers**")
                st.info("Consider hiring more assistants or cross-training existing teachers")
            
            underutilized = workload_df[workload_df['Utilization_%'] < 50]
            if not underutilized.empty:
                st.markdown("### 📊 Underutilized Teachers")
                st.write("These teachers have availability that could help fill gaps:")
                st.dataframe(underutilized, width="stretch", hide_index=True)
                st.info("💡 Consider adjusting their skills or availability to cover understaffed classes")

st.divider()
st.caption("Teacher Scheduling System | Powered by PuLP & Streamlit")
