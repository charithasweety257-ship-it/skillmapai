import difflib
import json
import os
from graphlib import TopologicalSorter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google import genai
import os


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="SkillMap AI",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# ENVIRONMENT / GEMINI CLIENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Change the model in your .env file without editing code, e.g.
# GEMINI_MODEL=gemini-2.5-flash
# Check Google's model docs for IDs your key can access.
MODEL_NAME = "gemini-3.8-flash"

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS_PATH = Path(__file__).parent / "skills.json"

# Alternate names students might type -> official skill name
ALIASES = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "scikitlearn": "Scikit-learn",
    "numpy": "NumPy",
    "genai": "Generative AI",
    "gen ai": "Generative AI",
    "llm": "Generative AI",
    "ds": "Data Science",
    "stats": "Statistics",
    "lang chain": "LangChain",
}


@st.cache_data
def read_skills_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_skills():
    try:
        return read_skills_file(str(SKILLS_PATH))
    except FileNotFoundError:
        st.error("skills.json was not found. Keep skills.json in the same folder as app.py.")
    except json.JSONDecodeError:
        st.error("skills.json contains invalid JSON.")
    return {}


skills_data = load_skills()

# lowercase name -> official name (built once)
SKILL_LOOKUP = {str(name).strip().lower(): name for name in skills_data}


# =========================================================
# SKILL HELPERS
# =========================================================

def find_skill(skill_name):
    """Return the official skill name for user text, or None."""
    key = str(skill_name).strip().lower()

    if key in SKILL_LOOKUP:
        return SKILL_LOOKUP[key]

    alias_target = ALIASES.get(key)
    if alias_target and alias_target.lower() in SKILL_LOOKUP:
        return SKILL_LOOKUP[alias_target.lower()]

    return None


def get_prerequisites(skill):
    """Return the official names of a skill's direct prerequisites."""
    value = skills_data.get(skill, [])

    if isinstance(value, dict):
        value = list(value.keys())
    elif not isinstance(value, list):
        return []

    result = []
    for item in value:
        official = find_skill(item)
        if official is not None:
            result.append(official)
    return result


def find_missing_prerequisites(completed_skills):
    """All direct and indirect prerequisites the student hasn't completed."""
    completed = set(completed_skills)
    seen = set()
    stack = list(completed_skills)

    while stack:
        current = stack.pop()
        for prerequisite in get_prerequisites(current):
            if prerequisite not in seen:
                seen.add(prerequisite)
                stack.append(prerequisite)

    return sorted(seen - completed)


def order_skills(missing_skills):
    """Order skills so each one comes after its prerequisites."""
    missing = set(missing_skills)
    graph = {
        skill: {p for p in get_prerequisites(skill) if p in missing}
        for skill in missing
    }

    try:
        # Sort first so the result is stable between runs
        graph = {k: graph[k] for k in sorted(graph)}
        return list(TopologicalSorter(graph).static_order())
    except Exception:
        # Cycle in skills.json: fall back to alphabetical
        return sorted(missing)


def unmatched_with_suggestions(input_lines):
    """Return [(typed_text, suggestion_or_None)] for unrecognized entries."""
    results = []
    for item in input_lines:
        if find_skill(item) is None:
            close = difflib.get_close_matches(
                item.strip().lower(), list(SKILL_LOOKUP.keys()), n=1, cutoff=0.6
            )
            suggestion = SKILL_LOOKUP[close[0]] if close else None
            results.append((item, suggestion))
    return results


# =========================================================
# SKILL GRAPH
# =========================================================

def dot_escape(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def create_graph(completed_skills, missing_skills):
    completed = set(completed_skills)
    all_skills = completed | set(missing_skills)

    lines = [
        "digraph SkillGraph {",
        "rankdir=LR;",
        'graph [bgcolor="transparent"];',
        'node [shape=box, style="rounded,filled", fontname="Arial"];',
    ]

    for skill in sorted(all_skills):
        fill = "#2ecc71" if skill in completed else "#f1c40f"
        name = dot_escape(skill)
        lines.append(f'"{name}" [label="{name}", fillcolor="{fill}"];')

    for skill in sorted(all_skills):
        for prerequisite in get_prerequisites(skill):
            if prerequisite in all_skills:
                lines.append(f'"{dot_escape(prerequisite)}" -> "{dot_escape(skill)}";')

    lines.append("}")
    return "\n".join(lines)


# =========================================================
# GEMINI AI RECOMMENDATION
# =========================================================

def generate_ai_recommendation(completed_skills, missing_skills):
    if client is None:
        return (
            "Gemini API key was not found.\n\n"
            "Please check that your .env file contains:\n\n"
            "`GEMINI_API_KEY=your_api_key`"
        )

    completed_text = ", ".join(completed_skills) or "None"
    missing_text = ", ".join(missing_skills) or "None"

    prompt = f"""
You are SkillMap AI, an AI learning advisor for a college student.

The student's completed skills are:
{completed_text}

The student's missing prerequisite skills are:
{missing_text}

Analyze this information and provide a short personalized learning
recommendation.

Include:
1. Recommended next skills
2. Why these skills are important
3. One beginner-friendly project idea
4. A short learning sequence

Keep the explanation simple and suitable for a college student.
Keep the whole answer under 200 words.

Only recommend skills related to the student's existing skills and
missing prerequisites. Do not invent unrelated skills.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        if response.text:
            return response.text

        return "Gemini returned an empty response."

    except Exception as error:
        return (
            "Unable to generate AI recommendation.\n\n"
            f"Error: {error}\n\n"
            f"Model used: `{MODEL_NAME}`. Please check your Gemini API key "
            "and model access."
        )


# =========================================================
# RESULTS DISPLAY
# =========================================================

def render_results(results):
    completed_skills = results["completed"]
    missing_skills = results["missing"]
    ordered_skills = results["ordered"]
    unmatched = results["unmatched"]

    # ---------------- Unrecognized entries ----------------
    if unmatched:
        st.divider()
        for item, suggestion in unmatched:
            hint = f" Did you mean **{suggestion}**?" if suggestion else ""
            st.warning(f"'{item}' wasn't found in the skills database.{hint}")

    # ---------------- Completed skills ----------------
    st.divider()
    st.subheader("✅ Your Completed Skills")

    if completed_skills:
        columns = st.columns(3)
        for index, skill in enumerate(completed_skills):
            with columns[index % 3]:
                st.success(f"✓ {skill}")
    else:
        st.warning("None of the entered skills were found in skills.json.")

    # ---------------- Missing skills ----------------
    st.divider()
    st.subheader("🎯 Missing Prerequisite Skills")

    if missing_skills:
        columns = st.columns(3)
        for index, skill in enumerate(missing_skills):
            with columns[index % 3]:
                st.warning(f"◆ {skill}")
    else:
        st.success("🎉 No missing prerequisites were found!")

    # ---------------- Skill graph ----------------
    st.divider()
    st.subheader("🕸️ Skill Graph")

    if completed_skills:
        st.graphviz_chart(
            create_graph(completed_skills, missing_skills),
            use_container_width=True,
        )
        st.caption("🟢 Green = Completed skill    🟡 Yellow = Missing prerequisite")
    else:
        st.info("Add valid skills to display the skill graph.")

    # ---------------- Roadmap ----------------
    st.divider()
    st.subheader("🛣️ Suggested Learning Roadmap")

    if ordered_skills:
        for index, skill in enumerate(ordered_skills, start=1):
            st.write(f"**{index}. {skill}**")

        st.info(
            "Complete the missing prerequisite skills in this order before "
            "progressing to your selected advanced skills."
        )
    else:
        st.success("You have completed the required prerequisite skills!")

    # ---------------- AI recommendation ----------------
    st.divider()
    st.subheader("🤖 AI-Powered Recommendation")

    if results["ai"] is None:
        st.info("Add at least one valid skill to get an AI recommendation.")
    else:
        st.markdown(results["ai"])


# =========================================================
# APPLICATION
# =========================================================

st.title("🧠 SkillMap AI")

st.write("AI-powered student skill gap analysis and learning roadmap")

st.write(
    "Enter the courses, skills, or projects you have already completed. "
    "SkillMap AI identifies prerequisite skills you may still need."
)

st.subheader("📚 Your Completed Courses / Projects")

st.caption("Enter one course, skill, or project per line.")

example_text = """Python
NumPy
Pandas
Statistics
Machine Learning"""

user_input = st.text_area(
    "Completed skills",
    value="",
    height=180,
    placeholder=example_text,
)

if st.button("🚀 Generate My Skill Map", use_container_width=True):

    input_lines = [line.strip() for line in user_input.splitlines() if line.strip()]

    if not input_lines:
        st.session_state.pop("results", None)
        st.warning("Please enter at least one completed course or skill.")

    else:
        completed_skills = []
        for item in input_lines:
            actual_skill = find_skill(item)
            if actual_skill is not None and actual_skill not in completed_skills:
                completed_skills.append(actual_skill)

        missing_skills = find_missing_prerequisites(completed_skills)
        ordered_skills = order_skills(missing_skills)

        ai_result = None
        if completed_skills:
            with st.spinner("Gemini AI is analyzing your skill profile..."):
                ai_result = generate_ai_recommendation(completed_skills, ordered_skills)

        # Saved so the output survives reruns caused by other widgets
        st.session_state["results"] = {
            "completed": completed_skills,
            "missing": missing_skills,
            "ordered": ordered_skills,
            "unmatched": unmatched_with_suggestions(input_lines),
            "ai": ai_result,
        }

if "results" in st.session_state:
    render_results(st.session_state["results"])


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("SkillMap AI • Student Skill Gap Analysis & Learning Roadmap")