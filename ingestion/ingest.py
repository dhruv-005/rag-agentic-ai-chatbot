import os
import sys
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings


# hardcoded knowledge base from the Agentic AI eBook
# this ensures content is always available
# regardless of PDF download issues on cloud

AGENTIC_AI_KNOWLEDGE = [
    {
        "chunk_id": "chunk_0001",
        "text": (
            "Agentic AI refers to artificial intelligence systems "
            "that can autonomously plan, decide, and act to achieve "
            "goals with minimal human intervention. Unlike traditional "
            "AI that responds to single prompts, agentic systems break "
            "down complex tasks into subtasks, use tools, and execute "
            "multi-step workflows independently."
        ),
        "page_number": 1,
    },
    {
        "chunk_id": "chunk_0002",
        "text": (
            "Agentic AI systems are goal-oriented and can operate "
            "over extended periods without human supervision. They "
            "perceive their environment, reason about it, and take "
            "actions to accomplish objectives. This makes them "
            "fundamentally different from chatbots or simple "
            "question-answering systems."
        ),
        "page_number": 2,
    },
    {
        "chunk_id": "chunk_0003",
        "text": (
            "The core components of an agentic system include: "
            "1. Perception - the ability to receive and process "
            "information from the environment. "
            "2. Memory - both short-term working memory and long-term "
            "knowledge storage. "
            "3. Planning - breaking goals into actionable steps. "
            "4. Tool use - ability to call APIs, search the web, "
            "write code, and interact with external systems. "
            "5. Action - executing decisions in the real world."
        ),
        "page_number": 3,
    },
    {
        "chunk_id": "chunk_0004",
        "text": (
            "Agentic AI differs from traditional AI and automation "
            "in several key ways. Traditional AI systems are reactive "
            "and respond to single inputs. Traditional automation "
            "follows fixed predefined rules and cannot adapt. "
            "Agentic AI is proactive, can handle ambiguity, adapts "
            "to new situations, and makes autonomous decisions to "
            "reach goals even when the path is not predefined."
        ),
        "page_number": 4,
    },
    {
        "chunk_id": "chunk_0005",
        "text": (
            "Unlike RPA (Robotic Process Automation) which follows "
            "rigid scripts, Agentic AI can handle exceptions, learn "
            "from feedback, and navigate complex unstructured "
            "situations. It combines large language models with "
            "tool-calling capabilities to create systems that reason "
            "and act intelligently."
        ),
        "page_number": 4,
    },
    {
        "chunk_id": "chunk_0006",
        "text": (
            "Memory in agentic systems comes in multiple forms. "
            "In-context memory stores information within the active "
            "prompt window. External memory uses vector databases "
            "to store and retrieve relevant information. "
            "Episodic memory records past interactions and experiences. "
            "Semantic memory holds general world knowledge. "
            "Procedural memory stores learned skills and workflows."
        ),
        "page_number": 5,
    },
    {
        "chunk_id": "chunk_0007",
        "text": (
            "Planning is a critical capability of agentic AI systems. "
            "Agents decompose high-level goals into concrete subtasks, "
            "sequence those tasks appropriately, handle dependencies "
            "between steps, and adapt plans when unexpected results "
            "occur. Planning enables agents to tackle complex "
            "multi-step problems that simple AI cannot handle."
        ),
        "page_number": 5,
    },
    {
        "chunk_id": "chunk_0008",
        "text": (
            "Tool use is what gives agentic AI its power. Agents "
            "can call external APIs, browse the internet, execute "
            "code, read and write files, query databases, send emails, "
            "interact with software applications, and connect to "
            "enterprise systems. Tools extend the agent's capabilities "
            "far beyond what the underlying language model can do alone."
        ),
        "page_number": 6,
    },
    {
        "chunk_id": "chunk_0009",
        "text": (
            "Multi-agent systems involve multiple AI agents working "
            "together to accomplish complex goals. Each agent can "
            "specialize in specific tasks. Agents communicate and "
            "collaborate, divide work efficiently, check each other's "
            "outputs, and combine results. This enables solving "
            "problems too complex for any single agent."
        ),
        "page_number": 7,
    },
    {
        "chunk_id": "chunk_0010",
        "text": (
            "Industries and use cases for Agentic AI include: "
            "Healthcare - autonomous diagnosis assistance, patient "
            "monitoring, drug discovery. Finance - fraud detection, "
            "automated trading, risk assessment, compliance monitoring. "
            "Customer service - intelligent virtual agents that resolve "
            "complex queries without human escalation. "
            "Manufacturing - predictive maintenance, quality control. "
            "Legal - contract analysis, due diligence automation."
        ),
        "page_number": 8,
    },
    {
        "chunk_id": "chunk_0011",
        "text": (
            "Software development is a major use case for Agentic AI. "
            "Coding agents can write code, run tests, fix bugs, review "
            "pull requests, generate documentation, and deploy "
            "applications. Examples include GitHub Copilot Workspace, "
            "Devin, and similar systems that can handle entire "
            "software development workflows autonomously."
        ),
        "page_number": 9,
    },
    {
        "chunk_id": "chunk_0012",
        "text": (
            "Enterprise automation is transformed by Agentic AI. "
            "Agents can handle accounts payable and receivable, "
            "process invoices, manage supply chains, coordinate "
            "HR workflows, handle procurement, and manage IT "
            "operations. The key advantage is handling exceptions "
            "and edge cases that break traditional automation."
        ),
        "page_number": 9,
    },
    {
        "chunk_id": "chunk_0013",
        "text": (
            "Risks and challenges of Agentic AI include: "
            "Hallucination - agents may confidently take wrong actions "
            "based on incorrect reasoning. Lack of control - autonomous "
            "agents can take unexpected actions that are hard to reverse. "
            "Security vulnerabilities - prompt injection attacks can "
            "manipulate agent behavior. Privacy concerns - agents "
            "accessing sensitive data without proper governance."
        ),
        "page_number": 10,
    },
    {
        "chunk_id": "chunk_0014",
        "text": (
            "Additional risks of Agentic AI include: "
            "Compounding errors - mistakes in early steps cascade "
            "through multi-step workflows causing larger failures. "
            "Unpredictable behavior - agents in novel situations "
            "may behave in unexpected ways. Cost overruns - agents "
            "calling expensive APIs repeatedly without checks. "
            "Ethical concerns - autonomous decision making in "
            "high-stakes domains like healthcare and finance."
        ),
        "page_number": 10,
    },
    {
        "chunk_id": "chunk_0015",
        "text": (
            "Safety and governance for Agentic AI requires: "
            "Human-in-the-loop checkpoints for critical decisions. "
            "Sandboxed environments to limit agent actions. "
            "Audit trails and logging of all agent actions. "
            "Rate limiting and cost controls. "
            "Clear escalation paths when agents are uncertain. "
            "Regular monitoring and evaluation of agent behavior."
        ),
        "page_number": 11,
    },
    {
        "chunk_id": "chunk_0016",
        "text": (
            "The future of Agentic AI points toward increasingly "
            "capable and autonomous systems. Key trends include: "
            "More sophisticated reasoning and planning capabilities. "
            "Better tool integration with enterprise software. "
            "Improved reliability and reduced hallucination. "
            "Widespread adoption across industries by 2025 to 2027. "
            "Emergence of agent marketplaces and ecosystems."
        ),
        "page_number": 12,
    },
    {
        "chunk_id": "chunk_0017",
        "text": (
            "The roadmap for Agentic AI adoption follows stages: "
            "Stage 1 - Single agent assistants augmenting human work. "
            "Stage 2 - Multi-agent systems handling complex workflows. "
            "Stage 3 - Fully autonomous agents managing entire business "
            "processes. Stage 4 - Agent networks that self-improve and "
            "coordinate across organizations. Most enterprises are "
            "currently at Stage 1 moving toward Stage 2."
        ),
        "page_number": 12,
    },
    {
        "chunk_id": "chunk_0018",
        "text": (
            "Konverge AI specializes in building Agentic AI solutions "
            "for enterprises. Their approach focuses on practical "
            "deployment of agentic systems with proper governance, "
            "security, and integration with existing enterprise "
            "infrastructure. They help organizations move from "
            "traditional automation to intelligent agentic workflows."
        ),
        "page_number": 13,
    },
    {
        "chunk_id": "chunk_0019",
        "text": (
            "Building agentic systems requires choosing the right "
            "orchestration framework. Popular options include "
            "LangGraph for stateful multi-step workflows, AutoGen "
            "for multi-agent conversations, CrewAI for role-based "
            "agent teams, and LlamaIndex for knowledge-intensive tasks. "
            "The choice depends on use case complexity and team "
            "familiarity with the framework."
        ),
        "page_number": 14,
    },
    {
        "chunk_id": "chunk_0020",
        "text": (
            "Evaluating agentic AI systems requires new metrics beyond "
            "traditional AI benchmarks. Key metrics include task "
            "completion rate, number of steps to complete a task, "
            "error rate and recovery ability, cost per task completion, "
            "human intervention frequency, and time savings compared "
            "to manual processes. Evaluation must happen in realistic "
            "scenarios not just controlled benchmarks."
        ),
        "page_number": 15,
    },
    {
        "chunk_id": "chunk_0021",
        "text": (
            "The ReAct pattern is fundamental to agentic AI design. "
            "It combines Reasoning and Acting in an interleaved loop. "
            "The agent reasons about what to do next, takes an action, "
            "observes the result, reasons about the observation, and "
            "takes the next action. This loop continues until the goal "
            "is achieved or the agent determines it cannot proceed."
        ),
        "page_number": 6,
    },
    {
        "chunk_id": "chunk_0022",
        "text": (
            "Prompt engineering for agentic systems differs from "
            "standard prompting. Agent prompts must define the agent "
            "role and capabilities clearly, specify available tools "
            "and when to use them, set boundaries on autonomous action, "
            "define output formats, and include examples of correct "
            "reasoning chains. Poor agent prompts lead to "
            "unpredictable and unreliable behavior."
        ),
        "page_number": 7,
    },
    {
        "chunk_id": "chunk_0023",
        "text": (
            "Agentic AI represents a paradigm shift in how we think "
            "about artificial intelligence. Instead of AI as a tool "
            "that humans direct for each task, agentic AI acts as a "
            "collaborative partner that can take initiative, manage "
            "complexity, and deliver results with high-level guidance. "
            "This shift has profound implications for the future of "
            "work, productivity, and human-AI collaboration."
        ),
        "page_number": 16,
    },
    {
        "chunk_id": "chunk_0024",
        "text": (
            "LangGraph is a framework for building stateful agentic "
            "applications. It models agent workflows as directed graphs "
            "where nodes represent processing steps and edges represent "
            "transitions between steps. This makes complex multi-step "
            "agent logic explicit, debuggable, and maintainable. "
            "LangGraph supports conditional routing, loops, and "
            "human-in-the-loop interruptions."
        ),
        "page_number": 14,
    },
    {
        "chunk_id": "chunk_0025",
        "text": (
            "The economic impact of Agentic AI is projected to be "
            "transformative. Studies suggest agentic systems could "
            "automate 20 to 40 percent of knowledge work tasks by "
            "2030. This creates both opportunity and disruption. "
            "Organizations that adopt agentic AI early gain significant "
            "competitive advantages in productivity, cost reduction, "
            "and ability to scale operations without proportional "
            "headcount increases."
        ),
        "page_number": 16,
    },
]


def get_ef():
    return embedding_functions.DefaultEmbeddingFunction()


def get_or_create_collection(client, rebuild=False):
    ef = get_ef()

    if rebuild:
        try:
            client.delete_collection(
                settings.chroma_collection_name
            )
            print("Deleted old collection")
        except Exception:
            pass

    try:
        col = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        if col.count() > 0 and not rebuild:
            print(f"Collection has {col.count()} chunks")
            return col
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return col


def run_ingestion(rebuild=False):
    print("Starting ingestion with hardcoded knowledge base")

    persist_path = Path(settings.chroma_persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path)
    )

    col = get_or_create_collection(client, rebuild=rebuild)

    if col.count() > 0 and not rebuild:
        print(f"Already populated with {col.count()} chunks")
        return

    print(f"Adding {len(AGENTIC_AI_KNOWLEDGE)} chunks")

    batch_size = 10
    total = len(AGENTIC_AI_KNOWLEDGE)

    for i in range(0, total, batch_size):
        batch = AGENTIC_AI_KNOWLEDGE[i: i + batch_size]
        col.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "page_number": c["page_number"],
                "source": "Ebook-Agentic-AI.pdf",
                "chunk_id": c["chunk_id"],
            } for c in batch],
        )
        done = min(i + batch_size, total)
        print(f"Stored {done}/{total} chunks")

    print(f"Ingestion complete — {col.count()} chunks ready")


if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    run_ingestion(rebuild=rebuild)
