
import os
import sys
import json
import re
import datetime
import PyPDF2
import time
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import streamlit as st
except ImportError:
    st = None

# Initialize OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key and st:
    try:
        if 'openai' in st.secrets and 'api_key' in st.secrets.openai:
            openai_key = st.secrets.openai.api_key
            os.environ["OPENAI_API_KEY"] = openai_key
    except Exception:
        pass

if not openai_key:
    print(
        "Warning: OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.",
        file=sys.stderr
    )

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# --- Constants ---
MAX_RETRIES = 3
MAX_WORKERS = 4
CHUNK_SIZE = 2500  

# --- LLM Initialization ---

def get_llm(model_name: str, temperature: float = 0.1):
    """Initializes and returns a ChatOpenAI instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OpenAI API key is not set.")
    
    # Compatible with langchain-openai>=0.1.20
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=90,
        max_retries=2
    )

# Initialize LLMs lazily - only when actually needed
llm_extractor = None
llm_assessor = None

def get_extractor_llm():
    """Get or create the extractor LLM instance."""
    global llm_extractor
    if llm_extractor is None:
        try:
            llm_extractor = get_llm("gpt-4o-mini")
        except EnvironmentError as e:
            if st:
                st.error(str(e))
            else:
                print(str(e), file=sys.stderr)
    return llm_extractor

def get_assessor_llm():
    """Get or create the assessor LLM instance."""
    global llm_assessor
    if llm_assessor is None:
        try:
            llm_assessor = get_llm("gpt-4o", 0.2)
        except EnvironmentError as e:
            if st:
                st.error(str(e))
            else:
                print(str(e), file=sys.stderr)
    return llm_assessor


# --- Prompt Templates ---

EXTRACTOR_PROMPT_TEMPLATE = """
You are an expert mentorship quality reviewer for Analytics Vidhya. Your role is to analyze a segment of a mentorship session transcript.
Your ONLY job is to EXTRACT evidence of guideline compliance and violations. DO NOT provide scores or high-level summaries.

EVALUATION SCOPE:
✓ Guideline compliance and adherence
✓ Professional conduct and communication style
✓ Session structure and time management
✓ Student engagement techniques
✓ Active listening and empathy
✗ Technical accuracy or depth of knowledge (DO NOT evaluate this)

TRANSCRIPT SEGMENT:
{chunk}

OFFICIAL MENTORSHIP GUIDELINES:
{guidelines}

EVALUATION INSTRUCTIONS:
1.  Focus on HOW the mentor communicates, not WHAT technical content they share.
2.  Extract direct quotes or specific behaviors as evidence.
3.  Identify the specific guideline related to each piece of evidence.
4.  Assign a severity (1-10) for violations only.

OUTPUT FORMAT - Complete this JSON structure. If a list is empty, return [].

{{
  "positive_behaviors": [
    {{
      "guideline": "Specific guideline followed (e.g., 'Active listening', 'Encouraging student participation')",
      "evidence": "Direct quote or specific behavior demonstrating this."
    }}
  ],
  "guideline_violations": [
    {{
      "guideline": "Specific guideline NOT followed (e.g., 'Interrupting student', 'Not checking for understanding')",
      "severity": <int 1-10, where 1=minor lapse, 10=critical violation>,
      "evidence": "Exact quote or specific behavior from transcript showing the violation."
    }}
  ],
  "key_mentor_topics": ["List of key topics the MENTOR explained in this segment"],
  "key_student_questions": ["List of key questions the STUDENT asked in this segment"]
}}
"""

FINAL_ASSESSOR_PROMPT_TEMPLATE = """
You are the Head of Mentorship Quality at Analytics Vidhya. You have received a set of observations from junior reviewers who analyzed chunks of a transcript.
Your job is to synthesize these raw observations into a single, comprehensive, and fair evaluation of the MENTOR's guideline adherence.

CONVERSATION METRICS:
{metrics}

AGGREGATED OBSERVATIONS (from all transcript chunks):

Positive Behaviors Observed:
{positive_behaviors}

Guideline Violations Observed:
{violations}

INSTRUCTIONS:
1.  Review all evidence provided. Look for patterns, not just isolated incidents.
2.  **Be fair and slightly lenient.** Start with a high score (e.g., 9/10) and only deduct points for clear, repeated, or severe violations based on the evidence. A few minor lapses should not heavily impact the score.
3.  Base your scores on the *balance and severity* of the evidence.
4.  Generate 2-4 key strengths based on *patterns* in the positive behaviors.
5.  **Generate a list of 3 to 5 `key_improvements` based on *patterns* in the violations.**
6.  **This is a strict requirement. You MUST provide a minimum of 3 improvement points and a maximum of 5.**
7.  If there are fewer than 3 clear violation patterns, identify 3 minor areas for refinement or "next-level" polish.
8.  If there are more than 5 violation patterns, synthesize them into the top 3-5 most impactful themes.
9.  Your justification (notes) for each score is critical.

OUTPUT FORMAT - Complete this JSON structure:

{{
  "overall_summary": "A detailed, 4-6 sentence qualitative paragraph. This summary must synthesize the mentor's performance against guidelines. It should start with an overall impression, then elaborate on the most significant strengths and the primary areas for improvement, connecting them to the specific evidence patterns.",
  "scores": {{
    "professionalism": <int 1-10>,
    "professionalism_notes": "Justification for the score based on evidence (e.g., tone, respect, punctuality).",
    "session_flow": <int 1-10>,
    "session_flow_notes": "Justification for the score (e.g., agenda setting, time management, logical progression).",
    "overall_guideline_compliance": <int 1-10>,
    "compliance_notes": "Justification for the score (e.g., active listening, engagement, checking understanding)."
  }},
  "key_strengths": [
    "LLM-generated strength based on positive_behaviors (e.g., 'Excellent use of active listening to confirm student's problems.')",
    "LLM-generated strength based on positive_behaviors (e.g., 'Effectively set the session agenda at the beginning.')"
  ],
  "key_improvements": [
    {{
      "title": "Specific, actionable improvement area (e.g., 'Consistent Check-ins for Understanding')",
      "suggestion": "A natural language explanation for the mentor. Explain *what* the issue was (based on violation patterns) and *how* they can improve it, using a supportive and professional tone. (e.g., 'After explaining a complex topic like Llama Parse, consider pausing to ask the student to re-explain it in their own words. This confirms understanding before moving on.')"
    }},
    {{
      "title": "Second Improvement Area",
      "suggestion": "A supportive and explanatory suggestion for the second area."
    }},
    {{
      "title": "Third Improvement Area",
      "suggestion": "A supportive and explanatory suggestion for the third area."
    }}
  ]
}}
"""

CHECKLIST_PROMPT_TEMPLATE = """
You are an auditor for Analytics Vidhya. Your task is to answer these specific questions based on the transcript.

**PRIMARY SOURCE OF TRUTH:**
FULL_TRANSCRIPT_TEXT:
{full_transcript_text}

---
SECONDARY SUMMARIES (Context only - DO NOT TRUST BLINDLY):
Positive Behaviors: {positive_behaviors}
Violations: {violations}
---

INSTRUCTIONS:
1.  **Scan the FULL_TRANSCRIPT_TEXT** to find the answer.
2.  **ANTI-HALLUCINATION RULE:** If you cannot find *explicit* evidence in the text, you MUST answer "UNCLEAR". Do not guess. Do not rely on the "Secondary Summaries" if they conflict with the transcript.
3.  Answer "YES", "NO", or "UNCLEAR".
4.  **Provide a COMPREHENSIVE, STRUCTURED EXPLANATION** for each answer.
    -   **Detailed Narrative:** The user wants "bigger summaries" that explain things fully. Do not be brief. Each explanation must be a substantial paragraph (4-5 sentences) followed by bullet points.
    -   **Structure:** Use **Bold Headers** for key points (e.g., **Roadmap realignment:**, **Short-term milestone:**).
    -   **Bullet Points:** Use bullet points to list specific evidence, questions asked, or topics covered.
    -   **Tone:** Professional, objective, and evidence-based.
    -   **Completeness:** Ensure every part of the question is addressed in the explanation. For example, if the question asks about summary, expectations, AND commitments, your explanation MUST cover all three explicitly.

**REFERENCE EXAMPLES (FOLLOW THIS STYLE EXACTLY):**

*Example 1: Challenges & Issues*
*Question:* "Did the mentor actively inquire about the student's current challenges (technical or otherwise) and help pin point specific issues or concerns?"
*Answer:* "YES"
*Explanation:* "Yes. The mentor actively inquired about the learner’s situation and framed concrete issues to address. Examples:
*   **Asked about current progress and time constraints:** “How long you’ve been doing this course…? I know you had some project work… what should I do to fast track?” and “What can I do in these two months to speed up development?”
*   **Probed the learner’s context and needs:** Inquired about the learner’s office use cases (RAG for customer feedback), and whether the roadmap needed updating.
*   **Sought concrete road-map adjustments:** Offered to reorder and tailor the learner’s courses, and suggested which modules to prioritize (e.g., LangGraph first, then advanced LangChain, and then other agents).
*   **Provided actionable steps:** Recommended specific tools (Wind SERP, VS Code-based tools), highlighted which chapters to skip or focus on, and outlined a concrete sequence to complete two to three courses quickly.
Overall, the mentor guided the learner with targeted questions and concrete, actionable recommendations to address current challenges and accelerate progress."

*Example 2: Commitments & Follow-up*
*Question:* "Did the mentor encourage the student to commit to learning goals (e.g., study hours, self-review) and create an actionable follow-up plan?"
*Answer:* "YES"
*Explanation:* "Yes. The mentor encouraged a concrete commitment to learning goals and provided an actionable follow-up plan. Key elements:
*   **Time and effort guidance:** Suggested spending hours per week (e.g., “five hours a week” or “20 hours a week” to speed up progress) and set expectations for coursework completion timeframe.
*   **Specific next steps:** Recommended finishing 2–3 more courses by the next session and focusing on the top 11 topics, with a reordered roadmap.
*   **Structured learning order:** Proposed a clear sequence (start with LangGraph/LangChain basics, then move to more advanced or alternative frameworks as needed).
*   **Regular check-ins:** Offered weekly 30-minute mentorship sessions as an ongoing accountability mechanism.
*   **Practical application:** Emphasized applying knowledge to real office use cases to reinforce learning and motivation.
*   **Roadmap action:** Offered to create/reorder the learner’s roadmap and commit to monitoring progress.
Overall, the guidance was aimed at a tangible, time-bound plan with ongoing mentorship to maintain accountability."

*Example 3: Summary & Expectations*
*Question:* "Did the mentor summarize the session, set expectations, and take clear commitments from the student on specific milestones?"
*Answer:* "YES"
*Explanation:* "Yes. The mentor summarized the session and set clear expectations, aiming to fast-track the learner’s progress. Key commitments and milestones included:
*   **Roadmap realignment:** The mentor proposed reordering the learner’s roadmap to prioritize practical, in-demand courses (LangChain, LangGraph, etc.) and to focus on building real projects.
*   **Short-term milestone:** The learner committed to completing 2–3 more courses by next week to accelerate progress.
*   **Ongoing guidance:** The mentor reinforced access to weekly, 30-minute on-demand mentorship sessions and emphasized applying knowledge to real office use cases.
*   **Clear next steps:** Finish the top prioritized 11 topics, then progress to more advanced material as planned.
Overall, the mentor provided a concise summary, set tangible milestones, and secured concrete commitments from the learner."

*Example 4: Background Knowledge*
*Question:* "Did the mentor appear well-informed about the student's background or ask relevant questions to understand it before offering guidance?"
*Answer:* "YES"
*Explanation:* "Yes. The mentor asked several relevant questions to understand the student’s background and needs before offering guidance. Key points:
*   **Asked about how long the student has been in the course, prior learning, and time availability.**
*   **Inquired about the student’s current role, employer, and use cases** (e.g., customer feedback, RAG, AI agents) to tailor advice.
*   **Confirmed which courses had been completed** and what the roadmap currently looked like.
*   **Verified practical constraints** (e.g., upcoming busy period, planned February/March timelines) to align the pace.
*   **Probed for specific goals** (fast-tracking learning, applying knowledge to real projects) and readiness to restart focused work.
Overall, the mentor demonstrated intent to understand the student’s background and objectives before guiding next steps."

*Example 5: Disruptions*
*Question:* "Were there any disruptions during the session, such as network issues, background noise, or the mentor not logging in on time?"
*Answer:* "NO" (or specific details if YES)
*Explanation:* "There were minor audio/clarity issues at the start: the mentor noted low volume and asked to confirm hearing.
*   There was a name/login mismatch for the learner (the mentor mentioned the login appeared as Rohi, later explained as the learner’s name not being updated due to a personal name issue).
*   No evidence of significant network outages or the mentor/logging in being delayed; the session proceeded and discussion continued normally.
*   A small UI/recording note about a roadmap display and a temporary bug in course ordering was mentioned, but it did not disrupt the session flow."

---

OUTPUT FORMAT - Complete this JSON structure:

{{
    "checklist": [
        {{
            "question": "Did mentor have camera feed with Virtual Background or Blur background?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. If NO, explain what happened. If YES, confirm it was on."
        }},
        {{
            "question": "Was there any network issues or background noise ?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Mention if there were any specific disruptions (e.g., 'A brief screen sharing hiccup around 02:20') or confirm 'No major disruptions reported'."
        }},
        {{
            "question": "Did mentor login on Time?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Cite specific timestamps or apologies for delay if present."
        }},
        {{
            "question": "Did mentor look like he/she knows the student's profile or have asked the same from student?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Describe how the mentor demonstrated knowledge of the student's background or what specific questions they asked to learn about it. Use bullet points if they asked multiple questions."
        }},
        {{
            "question": "Did mentor ask students what Challenges they are facing currently?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Quote the specific questions the mentor asked regarding challenges. Use bold headers for key points."
        }},
        {{
            "question": "Identify the specific issue/concern by discussing with student (If Technical Session)",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Summarize the technical discussion and the specific issue identified."
        }},
        {{
            "question": "Commitment taken from student (On their learning time, weekly review of their own progress)",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Specify exactly what the student committed to (hours per week, specific deadlines, etc.). Use bold headers for key commitments."
        }},
        {{
            "question": "Did the mentor summarize the session, set expectations, and take clear commitments from the student on specific milestones?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. **CRITICAL:** Check the END of the transcript. If the mentor did NOT explicitly summarize or ask for commitments, answer NO. Break this down into 'Summary', 'Expectations', and 'Commitments' using bold headers and each in separate bullet points."
        }},
        {{
            "question": "Did the mentor maintain a positive and professional tone throughout the session, without sharing negative remarks about AV or its courses?",
            "answer": "YES/NO/UNCLEAR",
            "explanation": "Detailed explanation. Confirm if the tone was professional and if any negative remarks were made."
        }}
    ]
}}
"""



# --- Core Logic Functions ---

@lru_cache(maxsize=32)
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF with caching to avoid repeated reads."""
    try:
        if not os.path.exists(pdf_path):
            return "Standard mentorship guidelines apply."
            
        mtime = os.path.getmtime(pdf_path)
        cache_key = f"{pdf_path}:{mtime}"
        
        # Simple cache implementation
        if not hasattr(extract_pdf_text, '_cache'):
            extract_pdf_text._cache = {}
        if cache_key in extract_pdf_text._cache:
            return extract_pdf_text._cache[cache_key]
            
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(
                page.extract_text() 
                for page in pdf_reader.pages 
                if page.extract_text()
            )
            
            extract_pdf_text._cache[cache_key] = text
            return text
    except Exception as e:
        if st:
            st.warning(f"Error reading PDF: {str(e)}")
        return "Standard mentorship guidelines apply."

def extract_json_from_text(text: str) -> str:
    """Extract JSON from text response, handling markdown code blocks."""
    match = re.search(r'```(json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(2)
    
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    
    print(f"Warning: Could not find JSON in text: {text[:200]}...")
    return "{}"

def chunk_text_intelligently(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks, trying to keep conversations intact."""
    if not text:
        return []
        
    lines = text.split('\n')
    if not lines:
        return []
        
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_len = len(line) + 1
        
        if (current_length + line_len > max_chars and current_chunk):
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_len
        else:
            current_chunk.append(line)
            current_length += line_len
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def analyze_conversation_metrics(transcript_text: str) -> dict:
    """Analyze conversation metrics from transcript."""
    lines = transcript_text.split('\n')
    
    speakers = {}
    for line in lines[:50]:
        if ':' in line:
            speaker = line.split(':', 1)[0].strip().lower()
            speakers[speaker] = speakers.get(speaker, 0) + 1
    
    sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
    
    mentor_name = "mentor"
    student_name = "student"
    
    if len(sorted_speakers) >= 2:
        mentor_name = sorted_speakers[0][0]
        student_name = sorted_speakers[1][0]
    elif len(sorted_speakers) == 1:
        mentor_name = sorted_speakers[0][0]

    mentor_words = 0
    student_words = 0
    
    for line in lines:
        line_lower = line.strip().lower()
        if line_lower.startswith(mentor_name + ':'):
            mentor_words += len(line.split(' ', 1)[-1].split())
        elif line_lower.startswith(student_name + ':'):
            student_words += len(line.split(' ', 1)[-1].split())

    total_words = mentor_words + student_words
    return {
        "mentor_talk_ratio": mentor_words / total_words if total_words else 0.5,
        "student_talk_ratio": student_words / total_words if total_words else 0.5,
        "total_exchanges": len(lines)
    }

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def analyze_single_chunk(args: Tuple[int, str, str]) -> Optional[dict]:
    """Process a single chunk of text with retry logic."""
    chunk_idx, chunk, guidelines = args
    
    try:
        llm = get_extractor_llm()
        if not llm:
            raise EnvironmentError("Extractor LLM not initialized.")
            
        prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
            chunk=chunk,
            guidelines=guidelines
        )
        
        response = llm.invoke(prompt)
        analysis_text = response.content if hasattr(response, 'content') else str(response)
        
        json_text = extract_json_from_text(analysis_text)
        analysis_data = json.loads(json_text)
        
        return {
            "chunk_index": chunk_idx,
            "analysis": analysis_data
        }
    except Exception as e:
        print(f"Error in chunk {chunk_idx}: {str(e)}")
        return {
            "chunk_index": chunk_idx,
            "analysis": {
                "positive_behaviors": [],
                "guideline_violations": [{
                    "guideline": "Analysis Error", 
                    "severity": 5, 
                    "evidence": f"Automated review failed for this segment: {str(e)[:150]}"
                }],
                "key_mentor_topics": [],
                "key_student_questions": []
            }
        }

def deep_analyze_chunks(chunks: List[str], guidelines: str) -> List[dict]:
    """Process multiple chunks in parallel."""
    results = []
    
    args_list = [(i, chunk, guidelines) for i, chunk in enumerate(chunks)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(analyze_single_chunk, args): args 
            for args in args_list
        }
        
        progress_bar = None
        if st:
            progress_bar = st.progress(0, "Analyzing transcript chunks...")
            
        for i, future in enumerate(as_completed(future_to_chunk)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing chunk future: {e}")
            
            if progress_bar:
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress, f"Analyzing transcript chunks... {i+1}/{len(chunks)}")
        
        if progress_bar:
            progress_bar.empty()
    
    results.sort(key=lambda x: x.get("chunk_index", 0))
    return results

def generate_final_assessment(chunk_analyses: List[dict], conversation_metrics: dict) -> dict:
    """Aggregate chunk data and call the final assessor LLM."""
    all_positive_behaviors = []
    all_violations = []

    for ca in (chunk_analyses or []):
        analysis = (ca or {}).get("analysis", {})
        all_positive_behaviors.extend(analysis.get("positive_behaviors", []) or [])
        all_violations.extend(analysis.get("guideline_violations", []) or [])

    try:
        llm = get_assessor_llm()
        if not llm:
            raise EnvironmentError("Assessor LLM not initialized.")
            
        prompt = FINAL_ASSESSOR_PROMPT_TEMPLATE.format(
            metrics=json.dumps(conversation_metrics, indent=2),
            positive_behaviors=json.dumps(all_positive_behaviors, indent=2),
            violations=json.dumps(all_violations, indent=2)
        )
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        json_text = extract_json_from_text(response_text)
        final_assessment = json.loads(json_text)
        
        final_assessment["raw_data"] = {
            "positive_behaviors": all_positive_behaviors,
            "violations": all_violations
        }
        
        scores = final_assessment.get("scores", {})
        avg_score_10 = (
            scores.get("professionalism", 5) + 
            scores.get("session_flow", 5) + 
            scores.get("overall_guideline_compliance", 5)
        ) / 3
        
        final_assessment["overall_score_100"] = round(avg_score_10 * 10, 1)

        return final_assessment
        
    except Exception as e:
        print(f"Error in final assessment: {str(e)}")
        return {
            "overall_summary": "Failed to generate final assessment.",
            "scores": {
                "professionalism": 0, "professionalism_notes": str(e),
                "session_flow": 0, "session_flow_notes": str(e),
                "overall_guideline_compliance": 0, "compliance_notes": str(e)
            },
            "key_strengths": [],
            "key_improvements": [{"title": "Assessment Failed", "suggestion": str(e)}],
            "overall_score_100": 0,
            "raw_data": {
                "positive_behaviors": all_positive_behaviors,
                "violations": all_violations
            }
        }

def generate_session_checklist(
    all_positive_behaviors: List[dict], 
    all_violations: List[dict], 
    full_transcript_text: str
) -> dict:
    """Generate a session checklist using LLM for decision making."""
    try:
        # Use Assessor LLM (GPT-4o) for better reasoning on the full transcript
        llm = get_assessor_llm()
        if not llm:
            raise EnvironmentError("Assessor LLM not initialized.")
            
        prompt = CHECKLIST_PROMPT_TEMPLATE.format(
            full_transcript_text=full_transcript_text,
            positive_behaviors=json.dumps(all_positive_behaviors, indent=2),
            violations=json.dumps(all_violations, indent=2)
        )
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        json_text = extract_json_from_text(response_text)
        checklist_data = json.loads(json_text)
        
        checklist_items = checklist_data.get('checklist', [])
        
        yes_count = sum(1 for item in checklist_items if item.get("answer", "").upper() == "YES")
        
        return {
            "checklist": checklist_items,
            "total_items": len(checklist_items),
            "passed_items": yes_count
        }
        
    except Exception as e:
        print(f"Error generating checklist with LLM: {str(e)}")
        default_checklist = [
            {"question": "Did mentor have camera feed with Virtual Background or Blur background?", "answer": "UNCLEAR", "explanation": f"Error: {str(e)}"},
            {"question": "Was there any network issues or background noise ?", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Did mentor login on Time?", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Did mentor look like he/she knows the student's profile or have asked the same from student?", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Did mentor ask students what Challenges they are facing currently?", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Identify the specific issue/concern by discussing with student (If Technical Session)", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Commitment taken from student (On their learning time, weekly review of their own progress)", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Did the mentor summarize the session, set expectations, and take clear commitments from the student on specific milestones?", "answer": "UNCLEAR", "explanation": "Error in evaluation"},
            {"question": "Did the mentor maintain a positive and professional tone throughout the session, without sharing negative remarks about AV or its courses?", "answer": "UNCLEAR", "explanation": "Error in evaluation"}]
        return {
            "checklist": default_checklist,
            "total_items": len(default_checklist),
            "passed_items": 0
        }

def generate_feedback_email(mentor_name: str, assessment: dict) -> str:
    """Generate feedback email for mentor with hardcoded strengths."""
    fixed_strengths = [
        "Demonstrated solid understanding of the topic.",
        "Maintained a helpful and professional attitude.",
        "Gave clear, practical technical advice."
    ]
    
    improvements = assessment.get("key_improvements", [])
    seen_titles = set()
    structured_issues = []
    
    for imp in improvements:
        title = imp.get("title", "").strip()
        if title and title.lower() not in seen_titles:
            structured_issues.append(imp)
            seen_titles.add(title.lower())
        if len(structured_issues) >= 5:
            break

    email = f"Hi {mentor_name},\n\n"
    email += "Thank you for your recent mentorship session with Analytics Vidhya. Below is focused feedback on your adherence to mentorship guidelines.\n\n"
    email += "✅ STRENGTHS\n"
    for s in fixed_strengths:
        email += f"• {s}\n"
    email += "\n🔧 AREAS FOR IMPROVEMENT\n"
    
    if structured_issues:
        for idx, issue in enumerate(structured_issues, 1):
            email += f"{idx}. {issue.get('title','')}\n"
            suggestion = issue.get('suggestion','')
            if suggestion:
                email += f"   - Suggestion: {suggestion}\n\n"
    else:
        email += "No significant compliance issues observed.\n"
    
    email += "\nPlease review these points to align future sessions with Analytics Vidhya's standards.\n\n"
    email += "Best regards,\nAnalytics Vidhya Mentorship Review Team"
    return email

def extract_mentor_name(transcript_data: Any, transcript_text: str) -> str:
    """Extract mentor name from transcript."""
    exclude_names = ['student', 'unknown', '', 'introduction']
    
    if isinstance(transcript_data, list):
        names = [
            item.get('speaker_name', '').strip() 
            for item in transcript_data 
            if isinstance(item, dict)
        ]
        mentor_candidates = [
            n for n in set(names) 
            if n and n.lower() not in exclude_names and len(n.split()) < 4
        ]
        if mentor_candidates:
            freq = {name: names.count(name) for name in mentor_candidates}
            if freq:
                return max(freq.items(), key=lambda x: x[1])[0]
    
    lines = transcript_text.strip().split('\n')[:20]
    candidates = {}
    for line in lines:
        if ':' in line:
            speaker = line.split(':', 1)[0].strip()
            if speaker and speaker.lower() not in exclude_names and len(speaker.split()) < 4:
                candidates[speaker] = candidates.get(speaker, 0) + 1
    if candidates:
        return max(candidates.items(), key=lambda x: x[1])[0]
    
    return "Mentor"

# --- Main Orchestrator ---

def process_transcript_enhanced(
    transcript_path: str, 
    guidelines_path: str = "Guidelines.pdf",
    session_date: Optional[str] = None,
    session_time: Optional[str] = None
) -> dict:
    """Main function to process transcript and generate comprehensive review."""
    
    # Verify LLMs can be initialized
    if not get_extractor_llm() or not get_assessor_llm():
        return {
            "success": False,
            "error": "LLM clients are not initialized. Check API key.",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        
        if isinstance(transcript_data, list):
            transcript_text = "\n".join(
                f"{item.get('speaker_name', 'Unknown')}: {item.get('sentence', '')}" 
                for item in transcript_data
            )
        else:
            transcript_text = transcript_data.get('text', str(transcript_data))
        
        guidelines = extract_pdf_text(guidelines_path)
        conversation_metrics = analyze_conversation_metrics(transcript_text)
        
        chunks = chunk_text_intelligently(transcript_text)
        chunk_analyses = deep_analyze_chunks(chunks, guidelines)
        
        if st: st.info("Synthesizing final assessment...")
        final_assessment = generate_final_assessment(chunk_analyses, conversation_metrics)
        
        if st: st.info("Generating session checklist...")
        session_checklist = generate_session_checklist(
            final_assessment.get("raw_data", {}).get("positive_behaviors", []),
            final_assessment.get("raw_data", {}).get("violations", []),
            transcript_text
        )
        
        if st: st.info("Drafting feedback email...")
        mentor_name = extract_mentor_name(transcript_data, transcript_text)
        email_content = generate_feedback_email(mentor_name, final_assessment)
        
        scores = final_assessment.get("scores", {})
        
        output = {
            "success": True,
            "mentor_name": mentor_name,
            "session_date": session_date or "N/A",
            "session_time": session_time or "N/A",
            "final_assessment": final_assessment,
            "feedback_email": email_content,
            "overall_session_summary": final_assessment.get("overall_summary", "No summary generated."),
            "session_checklist": session_checklist,
            "conversation_metrics": conversation_metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_score": final_assessment.get("overall_score_100", 0),
            "scores": {
                "professionalism": scores.get("professionalism", 0),
                "session_flow": scores.get("session_flow", 0),
                "guideline_compliance": scores.get("overall_guideline_compliance", 0),
            },
            "overall_guideline_assessment": {
                 "overall_score": final_assessment.get("overall_score_100", 0),
                 "professionalism": scores.get("professionalism", 0),
                 "session_flow": scores.get("session_flow", 0),
                 "overall_guideline_compliance": scores.get("overall_guideline_compliance", 0),
                 "detailed_scores": {
                    "professionalism": scores.get("professionalism", 0),
                    "session_flow": scores.get("session_flow", 0),
                    "overall_quality": scores.get("overall_guideline_compliance", 0),
                 },
                 "positive_behaviors": final_assessment.get("raw_data", {}).get("positive_behaviors", []),
                 "violations": final_assessment.get("raw_data", {}).get("violations", []),
                 "improvements": final_assessment.get("key_improvements", [])
            }
        }
        
        output_path = transcript_path.replace('.json','_guideline_review_v2.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Review Output saved at: {output_path}")
        print("\n==== EMAIL ====\n")
        print(email_content)
        
        return output
        
    except Exception as e:
        print(f"Error processing transcript: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process mentor review transcripts.')
    parser.add_argument('--transcript', type=str, 
                        default="Gunjan-Hense-Generative-AI-Pinnacle-Program-Mentorship-97129e00-0ce6.json",
                        help='Path to the transcript JSON file')
    parser.add_argument('--guidelines', type=str, default="Guidelines.pdf",
                        help='Path to the guidelines PDF file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.transcript):
        print(f"Error: Transcript file not found at {args.transcript}", file=sys.stderr)
        sys.exit(1)

    try:
        start_time = time.time()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        result = process_transcript_enhanced(
            args.transcript, 
            args.guidelines,
            session_date=date_str,
            session_time=time_str
        )
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        
        print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
        
        if not result.get("success"):
            print(f"Processing failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Unhandled Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
