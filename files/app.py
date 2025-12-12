import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Teacher Scheduling Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed file path - always use this file
EXCEL_FILE_PATH = 'teacher_scheduling_template.xlsx'

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
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
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
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

def load_data():
    """Load data from Excel file"""
    try:
        teachers = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Teachers')
        availability = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Teacher_Availability')
        skills = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Teacher_Skills')
        requirements = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Class_Requirements')
        return teachers, availability, skills, requirements, None
    except Exception as e:
        return None, None, None, None, str(e)

def save_data(teachers, availability, skills, requirements):
    """Save data back to Excel file"""
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
    """Run the optimization algorithm"""
    
    # Data preparation
    availability_full = availability.merge(teachers[['Teacher_Name', 'Teacher_Type']], on='Teacher_Name')
    skills_full = skills.merge(teachers[['Teacher_Name', 'Teacher_Type']], on='Teacher_Name')
    
    requirements['Class_ID'] = requirements.apply(
        lambda x: f"{x['Class_Type']}_{x['Day']}_{x['Time_Slot']}", axis=1
    )
    requirements['TimeSlot_ID'] = requirements.apply(
        lambda x: f"{x['Day']}_{x['Time_Slot']}", axis=1
    )
    availability_full['TimeSlot_ID'] = availability_full.apply(
        lambda x: f"{x['Day']}_{x['Time_Slot']}", axis=1
    )
    
    # Create optimization model
    model = LpProblem("Teacher_Scheduling", LpMaximize)
    
    teacher_names = teachers['Teacher_Name'].tolist()
    class_ids = requirements['Class_ID'].tolist()
    
    x = LpVariable.dicts("assign", [(t, c) for t in teacher_names for c in class_ids], cat='Binary')
    class_fully_staffed = LpVariable.dicts("fully_staffed", class_ids, cat='Binary')
    class_lead_shortage = LpVariable.dicts("lead_shortage", class_ids, lowBound=0, cat='Integer')
    class_assistant_shortage = LpVariable.dicts("assistant_shortage", class_ids, lowBound=0, cat='Integer')
    
    # Objective function
    model += (
        100 * lpSum([class_fully_staffed[c] for c in class_ids]) - 
        50 * lpSum([class_lead_shortage[c] for c in class_ids]) -
        30 * lpSum([class_assistant_shortage[c] for c in class_ids]),
        "Maximize_Coverage"
    )
    
    # Add constraints
    for idx, req in requirements.iterrows():
        class_id = req['Class_ID']
        time_slot = req['TimeSlot_ID']
        class_type = req['Class_Type']
        
        eligible_leads = []
        eligible_assistants = []
        
        for teacher_name in teacher_names:
            is_available = availability_full[
                (availability_full['Teacher_Name'] == teacher_name) &
                (availability_full['TimeSlot_ID'] == time_slot) &
                (availability_full['Available'] == 1)
            ].shape[0] > 0
            
            can_teach = skills_full[
                (skills_full['Teacher_Name'] == teacher_name) &
                (skills_full['Class_Type'] == class_type) &
                (skills_full['Can_Teach'] == 1)
            ].shape[0] > 0
            
            if is_available and can_teach:
                teacher_type = teachers[teachers['Teacher_Name'] == teacher_name]['Teacher_Type'].values[0]
                if teacher_type == 'Lead':
                    eligible_leads.append(teacher_name)
                else:
                    eligible_assistants.append(teacher_name)
        
        # Lead constraint
        if eligible_leads:
            leads_assigned = lpSum([x[t, class_id] for t in eligible_leads])
            model += leads_assigned + class_lead_shortage[class_id] >= req['Lead_Teachers_Required']
        else:
            model += class_lead_shortage[class_id] >= req['Lead_Teachers_Required']
        
        # Assistant constraint
        if eligible_assistants:
            assistants_assigned = lpSum([x[t, class_id] for t in eligible_assistants])
            model += assistants_assigned + class_assistant_shortage[class_id] >= req['Assistant_Teachers_Required']
        else:
            model += class_assistant_shortage[class_id] >= req['Assistant_Teachers_Required']
        
        # Fully staffed constraint
        if eligible_leads and eligible_assistants:
            total_required = req['Lead_Teachers_Required'] + req['Assistant_Teachers_Required']
            total_assigned = lpSum([x[t, class_id] for t in (eligible_leads + eligible_assistants)])
            model += total_assigned >= total_required * class_fully_staffed[class_id]
            model += class_fully_staffed[class_id] <= total_assigned / total_required
    
    # No double-booking
    for teacher_name in teacher_names:
        for time_slot in requirements['TimeSlot_ID'].unique():
            classes_at_time = requirements[requirements['TimeSlot_ID'] == time_slot]['Class_ID'].tolist()
            model += lpSum([x[teacher_name, c] for c in classes_at_time]) <= 1
    
    # Can only assign if available and qualified
    for teacher_name in teacher_names:
        for idx, req in requirements.iterrows():
            class_id = req['Class_ID']
            time_slot = req['TimeSlot_ID']
            class_type = req['Class_Type']
            
            is_available = availability_full[
                (availability_full['Teacher_Name'] == teacher_name) &
                (availability_full['TimeSlot_ID'] == time_slot) &
                (availability_full['Available'] == 1)
            ].shape[0] > 0
            
            can_teach = skills_full[
                (skills_full['Teacher_Name'] == teacher_name) &
                (skills_full['Class_Type'] == class_type) &
                (skills_full['Can_Teach'] == 1)
            ].shape[0] > 0
            
            if not (is_available and can_teach):
                model += x[teacher_name, class_id] == 0
    
    # Max hours constraint
    for teacher_name in teacher_names:
        max_hours = teachers[teachers['Teacher_Name'] == teacher_name]['Max_Hours_Per_Week'].values[0]
        model += lpSum([x[teacher_name, c] for c in class_ids]) <= max_hours
    
    # Solve
    model.solve(PULP_CBC_CMD(msg=0))
    
    if model.status != 1:
        return None, None, "Could not find optimal solution"
    
    # Build results
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
            'Day': req['Day'],
            'Time_Slot': req['Time_Slot'],
            'Class_Type': req['Class_Type'],
            'Children': req['Number_of_Children'],
            'Lead_Required': req['Lead_Teachers_Required'],
            'Lead_Assigned': ', '.join(assigned_leads) if assigned_leads else 'NONE',
            'Assistant_Required': req['Assistant_Teachers_Required'],
            'Assistant_Assigned': ', '.join(assigned_assistants) if assigned_assistants else 'NONE',
            'Status': status
        })
    
    schedule_df = pd.DataFrame(schedule_data)
    
    # Sort schedule
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    schedule_df['Day'] = pd.Categorical(schedule_df['Day'], categories=day_order, ordered=True)
    schedule_df = schedule_df.sort_values(['Day', 'Time_Slot'])
    
    # Teacher workload
    workload_data = []
    for teacher_name in teacher_names:
        classes_assigned = len(teacher_assignments.get(teacher_name, []))
        max_hours = teachers[teachers['Teacher_Name'] == teacher_name]['Max_Hours_Per_Week'].values[0]
        teacher_type = teachers[teachers['Teacher_Name'] == teacher_name]['Teacher_Type'].values[0]
        utilization = (classes_assigned / max_hours) * 100 if max_hours > 0 else 0
        
        workload_data.append({
            'Teacher_Name': teacher_name,
            'Teacher_Type': teacher_type,
            'Classes_Assigned': classes_assigned,
            'Max_Hours': max_hours,
            'Utilization_%': round(utilization, 1)
        })
    
    workload_df = pd.DataFrame(workload_data).sort_values('Classes_Assigned', ascending=False)
    
    return schedule_df, workload_df, None

# ==================== MAIN APP ====================
st.markdown('<h1 class="main-header">📚 Teacher Scheduling Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.info(f"📁 Using file: `{EXCEL_FILE_PATH}`")
    
    st.divider()
    
    # Optimize button
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
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
        st.warning("⚠️ Data has been modified. Re-run optimization to see updated schedule.")
    
    st.divider()
    
    st.header("📖 Quick Guide")
    st.markdown("""
    **Edit Data Tab:**
    - Modify teachers, availability, skills, and requirements
    - Click Save Changes when done
    
    **Run Optimization:**
    - Generates the best schedule
    - Balances workload
    - Identifies gaps
    
    **View Results:**
    - Weekly calendar view
    - Daily statistics
    - Teacher workload analysis
    """)

# Main content - Tabs
tab_edit, tab_schedule, tab_stats, tab_detailed, tab_workload, tab_issues = st.tabs([
    "✏️ Edit Data", 
    "📅 Weekly Schedule", 
    "📊 Daily Stats", 
    "📋 Detailed Schedule", 
    "👥 Teacher Workload", 
    "⚠️ Issues"
])

with tab_edit:
    st.header("✏️ Edit Schedule Data")
    st.info("💡 Use filters to find records, make changes, then click 'Save Changes' and 'Run Optimization' to update the schedule.")
    
    # Load current data
    teachers, availability, skills, requirements, error = load_data()
    
    if error:
        st.error(f"Error loading data: {error}")
    else:
        # ========== Teachers Section ==========
        st.subheader("👨‍🏫 Teachers")
        
        # Filters for Teachers
        with st.expander("🔍 Filter Teachers", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                filter_teacher_type = st.multiselect(
                    "Filter by Type",
                    options=teachers['Teacher_Type'].unique().tolist(),
                    default=teachers['Teacher_Type'].unique().tolist(),
                    key="filter_teacher_type"
                )
            with col2:
                filter_teacher_search = st.text_input(
                    "Search by Name",
                    "",
                    key="filter_teacher_search"
                )
        
        # Apply filters
        filtered_teachers = teachers[teachers['Teacher_Type'].isin(filter_teacher_type)]
        if filter_teacher_search:
            filtered_teachers = filtered_teachers[
                filtered_teachers['Teacher_Name'].str.contains(filter_teacher_search, case=False, na=False)
            ]
        
        st.caption(f"Showing {len(filtered_teachers)} of {len(teachers)} teachers")
        
        edited_teachers = st.data_editor(
            filtered_teachers,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Teacher_Name": st.column_config.TextColumn("Teacher Name", required=True),
                "Teacher_Type": st.column_config.SelectboxColumn("Type", options=["Lead", "Assistant"], required=True),
                "Max_Hours_Per_Week": st.column_config.NumberColumn("Max Hours/Week", min_value=1, max_value=80, required=True)
            },
            key="teachers_editor"
        )
        
        st.divider()
        
        # ========== Teacher Availability Section ==========
        st.subheader("📅 Teacher Availability")
        st.caption("Set when each teacher is available (1 = Available, 0 = Not Available)")
        
        # Filters for Availability
        with st.expander("🔍 Filter Availability", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_avail_teachers = st.multiselect(
                    "Filter by Teacher",
                    options=sorted(availability['Teacher_Name'].unique().tolist()),
                    default=availability['Teacher_Name'].unique().tolist(),
                    key="filter_avail_teachers"
                )
            with col2:
                filter_avail_days = st.multiselect(
                    "Filter by Day",
                    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    default=availability['Day'].unique().tolist(),
                    key="filter_avail_days"
                )
            with col3:
                filter_avail_slots = st.multiselect(
                    "Filter by Time Slot",
                    options=sorted(availability['Time_Slot'].unique().tolist()),
                    default=availability['Time_Slot'].unique().tolist(),
                    key="filter_avail_slots"
                )
        
        # Apply filters
        filtered_availability = availability[
            (availability['Teacher_Name'].isin(filter_avail_teachers)) &
            (availability['Day'].isin(filter_avail_days)) &
            (availability['Time_Slot'].isin(filter_avail_slots))
        ]
        
        st.caption(f"Showing {len(filtered_availability)} of {len(availability)} availability records")
        
        edited_availability = st.data_editor(
            filtered_availability,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Teacher_Name": st.column_config.TextColumn("Teacher Name", required=True),
                "Day": st.column_config.SelectboxColumn("Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], required=True),
                "Time_Slot": st.column_config.TextColumn("Time Slot", required=True),
                "Available": st.column_config.NumberColumn("Available (1=Yes, 0=No)", min_value=0, max_value=1, required=True)
            },
            key="availability_editor"
        )
        
        st.divider()
        
        # ========== Teacher Skills Section ==========
        st.subheader("🎓 Teacher Skills")
        st.caption("Set which age groups/classes each teacher can teach (1 = Can Teach, 0 = Cannot)")
        
        # Filters for Skills
        with st.expander("🔍 Filter Skills", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                filter_skills_teachers = st.multiselect(
                    "Filter by Teacher",
                    options=sorted(skills['Teacher_Name'].unique().tolist()),
                    default=skills['Teacher_Name'].unique().tolist(),
                    key="filter_skills_teachers"
                )
            with col2:
                filter_skills_classes = st.multiselect(
                    "Filter by Class Type",
                    options=sorted(skills['Class_Type'].unique().tolist()),
                    default=skills['Class_Type'].unique().tolist(),
                    key="filter_skills_classes"
                )
        
        # Apply filters
        filtered_skills = skills[
            (skills['Teacher_Name'].isin(filter_skills_teachers)) &
            (skills['Class_Type'].isin(filter_skills_classes))
        ]
        
        st.caption(f"Showing {len(filtered_skills)} of {len(skills)} skill records")
        
        edited_skills = st.data_editor(
            filtered_skills,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Teacher_Name": st.column_config.TextColumn("Teacher Name", required=True),
                "Class_Type": st.column_config.TextColumn("Class Type", required=True),
                "Can_Teach": st.column_config.NumberColumn("Can Teach (1=Yes, 0=No)", min_value=0, max_value=1, required=True)
            },
            key="skills_editor"
        )
        
        st.divider()
        
        # ========== Class Requirements Section ==========
        st.subheader("📚 Class Requirements")
        st.caption("Define the classes you need to staff")
        
        # Filters for Requirements
        with st.expander("🔍 Filter Class Requirements", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_req_classes = st.multiselect(
                    "Filter by Class Type",
                    options=sorted(requirements['Class_Type'].unique().tolist()),
                    default=requirements['Class_Type'].unique().tolist(),
                    key="filter_req_classes"
                )
            with col2:
                filter_req_days = st.multiselect(
                    "Filter by Day",
                    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    default=requirements['Day'].unique().tolist(),
                    key="filter_req_days"
                )
            with col3:
                filter_req_slots = st.multiselect(
                    "Filter by Time Slot",
                    options=sorted(requirements['Time_Slot'].unique().tolist()),
                    default=requirements['Time_Slot'].unique().tolist(),
                    key="filter_req_slots"
                )
        
        # Apply filters
        filtered_requirements = requirements[
            (requirements['Class_Type'].isin(filter_req_classes)) &
            (requirements['Day'].isin(filter_req_days)) &
            (requirements['Time_Slot'].isin(filter_req_slots))
        ]
        
        st.caption(f"Showing {len(filtered_requirements)} of {len(requirements)} class requirements")
        
        edited_requirements = st.data_editor(
            filtered_requirements,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Class_Type": st.column_config.TextColumn("Class Type", required=True),
                "Day": st.column_config.SelectboxColumn("Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], required=True),
                "Time_Slot": st.column_config.TextColumn("Time Slot", required=True),
                "Number_of_Children": st.column_config.NumberColumn("# Children", min_value=1, required=True),
                "Teacher_Child_Ratio": st.column_config.TextColumn("Ratio (e.g., 1:10)", required=True),
                "Lead_Teachers_Required": st.column_config.NumberColumn("Lead Teachers", min_value=0, required=True),
                "Assistant_Teachers_Required": st.column_config.NumberColumn("Assistant Teachers", min_value=0, required=True)
            },
            key="requirements_editor"
        )
        
        st.divider()
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                # Merge filtered edited data with unfiltered data
                # For teachers
                teachers_to_save = teachers[~teachers['Teacher_Type'].isin(filter_teacher_type)]
                if filter_teacher_search:
                    teachers_to_save = pd.concat([
                        teachers_to_save,
                        teachers[~teachers['Teacher_Name'].str.contains(filter_teacher_search, case=False, na=False)]
                    ]).drop_duplicates()
                teachers_to_save = pd.concat([teachers_to_save, edited_teachers]).drop_duplicates(subset=['Teacher_Name'], keep='last')
                
                # For availability
                availability_to_save = availability[
                    ~(availability['Teacher_Name'].isin(filter_avail_teachers)) |
                    ~(availability['Day'].isin(filter_avail_days)) |
                    ~(availability['Time_Slot'].isin(filter_avail_slots))
                ]
                availability_to_save = pd.concat([availability_to_save, edited_availability]).drop_duplicates()
                
                # For skills
                skills_to_save = skills[
                    ~(skills['Teacher_Name'].isin(filter_skills_teachers)) |
                    ~(skills['Class_Type'].isin(filter_skills_classes))
                ]
                skills_to_save = pd.concat([skills_to_save, edited_skills]).drop_duplicates()
                
                # For requirements
                requirements_to_save = requirements[
                    ~(requirements['Class_Type'].isin(filter_req_classes)) |
                    ~(requirements['Day'].isin(filter_req_days)) |
                    ~(requirements['Time_Slot'].isin(filter_req_slots))
                ]
                requirements_to_save = pd.concat([requirements_to_save, edited_requirements]).drop_duplicates()
                
                success, save_error = save_data(teachers_to_save, availability_to_save, skills_to_save, requirements_to_save)
                if success:
                    st.session_state.data_modified = True
                    st.success("✅ Changes saved successfully! Click 'Run Optimization' in the sidebar to update the schedule.")
                else:
                    st.error(f"Error saving: {save_error}")
        
        with col2:
            if st.button("🔄 Reload Data", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Filters", use_container_width=True):
                # This will reset all filters by rerunning
                for key in st.session_state.keys():
                    if key.startswith('filter_'):
                        del st.session_state[key]
                st.rerun()

# Show data summary even if optimization hasn't run
if not st.session_state.optimization_run:
    with tab_schedule:
        st.info("👆 Click 'Run Optimization' in the sidebar to generate the schedule")
        
        # Show sample data if available
        try:
            teachers, availability, skills, requirements, error = load_data()
            if not error:
                st.subheader("📊 Current Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Teachers", len(teachers))
                with col2:
                    st.metric("Classes", len(requirements))
                with col3:
                    st.metric("Lead Teachers", len(teachers[teachers['Teacher_Type'] == 'Lead']))
                with col4:
                    st.metric("Assistant Teachers", len(teachers[teachers['Teacher_Type'] == 'Assistant']))
        except:
            pass
else:
    schedule_df = st.session_state.schedule_df
    workload_df = st.session_state.workload_df
    
    # Summary metrics
    total_classes = len(schedule_df)
    fully_staffed = len(schedule_df[schedule_df['Status'] == 'Fully Staffed'])
    understaffed = total_classes - fully_staffed
    
    # Show metrics in all relevant tabs
    for current_tab in [tab_schedule, tab_stats, tab_detailed, tab_workload, tab_issues]:
        with current_tab:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Classes", total_classes)
            with col2:
                st.metric("✅ Fully Staffed", fully_staffed, delta=f"{(fully_staffed/total_classes)*100:.0f}%")
            with col3:
                st.metric("⚠️ Understaffed", understaffed, delta=f"-{(understaffed/total_classes)*100:.0f}%" if understaffed > 0 else "0%")
            with col4:
                avg_util = workload_df['Utilization_%'].mean()
                st.metric("Avg Teacher Utilization", f"{avg_util:.1f}%")
            
            st.divider()
            break
    
    with tab_schedule:
        st.subheader("Weekly Calendar View")
        
        # Get unique days from schedule
        days = schedule_df['Day'].cat.categories.tolist()
        time_slots = sorted(schedule_df['Time_Slot'].unique())
        
        for time_slot in time_slots:
            st.markdown(f"### ⏰ {time_slot}")
            cols = st.columns(len(days))
            
            for i, day in enumerate(days):
                with cols[i]:
                    st.markdown(f"**{day}**")
                    
                    classes_at_slot = schedule_df[
                        (schedule_df['Day'] == day) & 
                        (schedule_df['Time_Slot'] == time_slot)
                    ]
                    
                    if not classes_at_slot.empty:
                        for _, cls in classes_at_slot.iterrows():
                            if cls['Status'] == 'Fully Staffed':
                                box_class = 'success-box'
                                icon = '✅'
                            else:
                                box_class = 'warning-box'
                                icon = '⚠️'
                            
                            # Age group emojis
                            if cls['Class_Type'] == 'Toddler':
                                age_icon = '👶'
                            elif cls['Class_Type'] == 'Pre-K':
                                age_icon = '🧒'
                            elif cls['Class_Type'] == 'Elementary':
                                age_icon = '👦'
                            elif cls['Class_Type'] == 'Infant':
                                age_icon = '🍼'
                            else:
                                age_icon = '📚'
                            
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
        st.subheader("📊 Daily Statistics - All Classes")
        
        days = schedule_df['Day'].cat.categories.tolist()
        
        for day in days:
            st.markdown(f"### 📅 {day}")
            
            day_classes = schedule_df[schedule_df['Day'] == day]
            
            if day_classes.empty:
                st.info(f"No classes scheduled on {day}")
                continue
            
            # Summary metrics for the day
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Classes", len(day_classes))
            with col2:
                total_children = day_classes['Children'].sum()
                st.metric("Total Children", total_children)
            with col3:
                staffed = len(day_classes[day_classes['Status'] == 'Fully Staffed'])
                st.metric("✅ Fully Staffed", staffed)
            with col4:
                understaffed_day = len(day_classes[day_classes['Status'] != 'Fully Staffed'])
                st.metric("⚠️ Understaffed", understaffed_day)
            
            # Breakdown by class type
            st.markdown("#### Classes by Age Group")
            
            class_types = day_classes['Class_Type'].unique()
            type_cols = st.columns(len(class_types) if len(class_types) <= 5 else 5)
            
            for idx, class_type in enumerate(class_types):
                col_idx = idx % 5
                with type_cols[col_idx]:
                    type_classes = day_classes[day_classes['Class_Type'] == class_type]
                    
                    if class_type == 'Toddler':
                        icon = '👶'
                    elif class_type == 'Pre-K':
                        icon = '🧒'
                    elif class_type == 'Elementary':
                        icon = '👦'
                    elif class_type == 'Infant':
                        icon = '🍼'
                    else:
                        icon = '📚'
                    
                    st.markdown(f"**{icon} {class_type}**")
                    
                    if not type_classes.empty:
                        st.write(f"Classes: {len(type_classes)}")
                        st.write(f"Children: {type_classes['Children'].sum()}")
                        
                        # Count teachers needed vs assigned
                        total_leads_needed = type_classes['Lead_Required'].sum()
                        total_assists_needed = type_classes['Assistant_Required'].sum()
                        
                        leads_assigned = sum(1 for x in type_classes['Lead_Assigned'] if x != 'NONE')
                        assists_count = sum(len(x.split(', ')) for x in type_classes['Assistant_Assigned'] if x != 'NONE')
                        
                        st.write(f"Lead: {leads_assigned}/{total_leads_needed}")
                        st.write(f"Assist: {assists_count}/{total_assists_needed}")
                        
                        # Status indicator
                        fully_staffed_type = len(type_classes[type_classes['Status'] == 'Fully Staffed'])
                        if fully_staffed_type == len(type_classes):
                            st.success("✅ All staffed")
                        else:
                            st.warning(f"⚠️ {len(type_classes) - fully_staffed_type} understaffed")
            
            # Detailed schedule for the day
            with st.expander(f"View Detailed {day} Schedule"):
                st.dataframe(
                    day_classes[['Time_Slot', 'Class_Type', 'Children', 'Lead_Assigned', 'Assistant_Assigned', 'Status']],
                    use_container_width=True,
                    hide_index=True
                )
            
            st.divider()
        
        # Weekly summary chart
        st.markdown("### 📈 Weekly Overview")
        
        # Children per day chart
        daily_children = schedule_df.groupby('Day')['Children'].sum()
        daily_classes = schedule_df.groupby('Day').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure(data=[
                go.Bar(x=daily_children.index.tolist(), y=daily_children.values, marker_color='#1f77b4')
            ])
            fig1.update_layout(
                title='Total Children per Day',
                xaxis_title='Day',
                yaxis_title='Number of Children',
                height=300
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = go.Figure(data=[
                go.Bar(x=daily_classes.index.tolist(), y=daily_classes.values, marker_color='#ff7f0e')
            ])
            fig2.update_layout(
                title='Total Classes per Day',
                xaxis_title='Day',
                yaxis_title='Number of Classes',
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Class type distribution
        st.markdown("### 🎯 Age Group Distribution")
        
        class_type_counts = schedule_df.groupby('Class_Type').agg({
            'Class_Type': 'count',
            'Children': 'sum'
        }).rename(columns={'Class_Type': 'Classes'})
        
        fig3 = go.Figure(data=[
            go.Bar(name='Classes', x=class_type_counts.index, y=class_type_counts['Classes'], marker_color='#2ca02c'),
            go.Bar(name='Children', x=class_type_counts.index, y=class_type_counts['Children'], marker_color='#d62728')
        ])
        fig3.update_layout(
            title='Classes and Children by Age Group',
            xaxis_title='Age Group',
            yaxis_title='Count',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab_detailed:
        st.subheader("Detailed Schedule")
        
        days = schedule_df['Day'].cat.categories.tolist()
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_days = st.multiselect("Filter by Day", days, default=days)
        with col2:
            age_groups = schedule_df['Class_Type'].unique().tolist()
            selected_age_groups = st.multiselect("Filter by Age Group", age_groups, default=age_groups)
        with col3:
            selected_status = st.multiselect("Filter by Status", 
                                            schedule_df['Status'].unique(), 
                                            default=schedule_df['Status'].unique())
        
        filtered_schedule = schedule_df[
            (schedule_df['Day'].isin(selected_days)) & 
            (schedule_df['Class_Type'].isin(selected_age_groups)) &
            (schedule_df['Status'].isin(selected_status))
        ]
        
        st.dataframe(filtered_schedule, use_container_width=True, hide_index=True)
        
        # Download button
        csv = filtered_schedule.to_csv(index=False)
        st.download_button(
            label="📥 Download Schedule as CSV",
            data=csv,
            file_name="teacher_schedule.csv",
            mime="text/csv"
        )
    
    with tab_workload:
        st.subheader("Teacher Workload Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Workload chart
            fig = px.bar(workload_df, 
                        x='Teacher_Name', 
                        y='Utilization_%',
                        color='Teacher_Type',
                        title='Teacher Utilization',
                        labels={'Utilization_%': 'Utilization (%)', 'Teacher_Name': 'Teacher'},
                        color_discrete_map={'Lead': '#1f77b4', 'Assistant': '#ff7f0e'})
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Max Capacity")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Summary Stats")
            st.metric("Most Utilized", 
                     workload_df.iloc[0]['Teacher_Name'],
                     f"{workload_df.iloc[0]['Utilization_%']}%")
            st.metric("Least Utilized", 
                     workload_df.iloc[-1]['Teacher_Name'],
                     f"{workload_df.iloc[-1]['Utilization_%']}%")
            
            underutilized = workload_df[workload_df['Utilization_%'] < 50]
            st.metric("Underutilized (<50%)", len(underutilized))
        
        st.divider()
        
        # Detailed workload table
        st.dataframe(workload_df, use_container_width=True, hide_index=True)
    
    with tab_issues:
        st.subheader("Issues & Recommendations")
        
        understaffed_df = schedule_df[schedule_df['Status'] != 'Fully Staffed']
        
        if understaffed_df.empty:
            st.success("🎉 Perfect! All classes are fully staffed!")
        else:
            st.warning(f"⚠️ {len(understaffed_df)} classes need attention")
            
            # Show understaffed classes
            st.markdown("### Understaffed Classes")
            st.dataframe(understaffed_df, use_container_width=True, hide_index=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            missing_leads = understaffed_df[understaffed_df['Status'].str.contains('Lead', na=False)]
            missing_assistants = understaffed_df[understaffed_df['Status'].str.contains('Assistant', na=False)]
            
            if not missing_leads.empty:
                st.markdown(f"🔴 **{len(missing_leads)} classes missing Lead Teachers**")
                st.info("Consider hiring more lead teachers or adjusting lead teacher availability")
            
            if not missing_assistants.empty:
                st.markdown(f"🟡 **{len(missing_assistants)} classes missing Assistant Teachers**")
                st.info("Consider hiring more assistants or cross-training existing teachers")
            
            # Show underutilized teachers
            underutilized = workload_df[workload_df['Utilization_%'] < 50]
            if not underutilized.empty:
                st.markdown("### 📊 Underutilized Teachers")
                st.write("These teachers have availability that could help fill gaps:")
                st.dataframe(underutilized, use_container_width=True, hide_index=True)
                st.info("💡 Consider adjusting their skills or availability to cover understaffed classes")

# Footer
st.divider()
st.caption("Teacher Scheduling Optimization System | Powered by PuLP & Streamlit")
