import os, re, math, datetime, functools

# Suppress HuggingFace warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

from typing import TypedDict, List, Optional
from dotenv import load_dotenv

load_dotenv()  # loads .env file before anything else reads env vars

# ─── Third-party imports ───────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Collection
from chromadb.api.types import Metadata

# ─── Configuration ─────────────────────────────────────────────────────────────
# os.getenv() returns str | None — use `or` fallback so it's always a str,
# then wrap in SecretStr() which is what ChatGroq's api_key parameter requires.
_raw_api_key = os.getenv("GROQ_API_KEY") or "your_groq_api_key_here"
GROQ_API_KEY = SecretStr(_raw_api_key)

MODEL_NAME       = "llama-3.1-8b-instant"
EMBED_MODEL      = "all-MiniLM-L6-v2"   # Real semantic embeddings
MAX_WINDOW       = 6              # Sliding window — keep last 6 messages
MAX_EVAL_RETRIES = 2              # Self-reflection eval retries
FAITH_THRESHOLD  = 0.7            # Faithfulness gate
SUPPORT_EMAIL    = "support@physicstudy.edu"
TOP_K            = 3              # Number of ChromaDB chunks to retrieve

# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    student_name: Optional[str]
    topic_asked:  Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — 12 Physics documents (100-400 words each, one topic each)
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "id": "doc_001", "topic": "Newtons Laws of Motion",
        "text": """
Newton's Laws of Motion are three fundamental principles describing the relationship between a body and forces acting upon it.

First Law (Law of Inertia): An object at rest stays at rest, and an object in motion stays in motion with the same speed and direction, unless acted upon by a net external force.

Second Law (Law of Acceleration): The net force on an object equals the product of its mass and acceleration.
Formula: F = ma
Where F = net force (Newtons, N), m = mass (kg), a = acceleration (m/s^2).
Rearranged: a = F/m (acceleration increases with force, decreases with mass).

Third Law (Action-Reaction): For every action there is an equal and opposite reaction. When object A exerts force on B, B exerts equal and opposite force on A.

Key concepts:
- Inertia: tendency to resist change in motion. Mass measures inertia.
- Unit of force: 1 Newton = 1 kg*m/s^2
- These laws apply in inertial (non-accelerating) reference frames.
- Net force = vector sum of all forces on the body.

Example: A 10 kg box pushed with 50 N net force accelerates at a = 50/10 = 5 m/s^2.
Example of Third Law: When you push against a wall, the wall pushes back on you with equal force.
        """
    },
    {
        "id": "doc_002", "topic": "Work Energy and Power",
        "text": """
Work, Energy, and Power describe how forces move objects and transfer energy.

Work (W): Done when force causes displacement. W = F*d*cos(theta)
Where F = force (N), d = displacement (m), theta = angle between force and displacement.
Unit: Joule (J). If force and displacement are parallel (theta=0): W = F*d.

Kinetic Energy (KE): Energy of motion. KE = (1/2)mv^2
Where m = mass (kg), v = velocity (m/s). Unit: Joule (J).

Potential Energy (PE): Stored energy due to position.
Gravitational PE = mgh, where g = 9.8 m/s^2, h = height above reference.
Elastic PE = (1/2)kx^2, where k = spring constant, x = compression/extension.

Work-Energy Theorem: Net work done = change in kinetic energy.
W_net = delta_KE = KE_final - KE_initial

Conservation of Mechanical Energy: Without friction, KE1 + PE1 = KE2 + PE2.
Total mechanical energy (KE + PE) remains constant.

Power (P): Rate of doing work.
P = W/t = F*v
Unit: Watt (W) = J/s. 1 horsepower = 746 W.

Example: Ball of mass 2 kg dropped from 5 m: KE gained = mgh = 2*9.8*5 = 98 J.
Example: Machine doing 500 J of work in 10 s has power = 500/10 = 50 W.
        """
    },
    {
        "id": "doc_003", "topic": "Laws of Thermodynamics",
        "text": """
The Laws of Thermodynamics govern all energy transformations involving heat and work.

Zeroth Law: If systems A and B are each in thermal equilibrium with system C, then A and B are in thermal equilibrium with each other. This defines temperature.

First Law (Conservation of Energy): Energy cannot be created or destroyed.
delta_U = Q - W
Where delta_U = change in internal energy, Q = heat added to system, W = work done BY the system.

Second Law: Heat flows naturally from hot to cold. No heat engine can be 100% efficient.
Entropy (S) = measure of disorder. Entropy of an isolated system never decreases: delta_S >= 0.

Third Law: As temperature approaches absolute zero (0 K = -273.15 C), entropy of a perfect crystal approaches 0. Absolute zero is theoretically unattainable.

Key process types:
- Isothermal (constant T): delta_U = 0, so Q = W.
- Adiabatic (no heat exchange): Q = 0, so delta_U = -W.
- Isobaric (constant pressure): W = P*delta_V.
- Isochoric (constant volume): W = 0, so delta_U = Q.

Carnot Efficiency: Maximum efficiency of any heat engine = 1 - (T_cold/T_hot), where temperatures are in Kelvin.
        """
    },
    {
        "id": "doc_004", "topic": "Coulombs Law and Electric Field",
        "text": """
Electrostatics describes forces between stationary electric charges.

Coulomb's Law: Electrostatic force between two point charges.
F = k*q1*q2 / r^2
Where k = 9e9 N*m^2/C^2 (Coulomb's constant), q1, q2 = charges (Coulombs), r = distance (m).
Like charges repel; unlike charges attract.

Electric Field (E): Force per unit positive test charge.
E = F/q_test = k*Q/r^2
Unit: N/C or V/m. Direction: away from positive charge, toward negative charge.

Electric Potential (V): Work done per unit charge from infinity to a point.
V = k*Q/r. Unit: Volt (V).
Relationship: E = -dV/dr (field points in direction of decreasing potential).

Superposition Principle: Total force (or field) = vector sum of individual contributions.

Gauss's Law: Total electric flux through a closed surface = Q_enclosed/epsilon_0.
Phi_E = Q_enc/epsilon_0, where epsilon_0 = 8.85e-12 C^2/(N*m^2).

Electric Flux: Phi = E*A*cos(theta), where A = area of surface.

Example: Two charges of +2 uC separated by 0.1 m. F = 9e9*(2e-6)^2/(0.1)^2 = 3.6 N (repulsive).
        """
    },
    {
        "id": "doc_005", "topic": "Simple Harmonic Motion",
        "text": """
Simple Harmonic Motion (SHM) is periodic motion where restoring force is proportional to displacement from equilibrium and directed toward it.

Condition: F = -kx (Hooke's Law). Negative sign = restoring force opposes displacement.

Key equations:
- Displacement: x(t) = A*cos(omega*t + phi), A = amplitude, omega = angular frequency, phi = initial phase.
- Velocity: v(t) = -A*omega*sin(omega*t + phi). Maximum velocity v_max = A*omega (at equilibrium, x=0).
- Acceleration: a(t) = -A*omega^2*cos(omega*t + phi). Maximum acceleration = A*omega^2 (at extremes).
- Angular frequency: omega = sqrt(k/m) for spring-mass; omega = sqrt(g/L) for simple pendulum.
- Time period: T = 2*pi/omega = 2*pi*sqrt(m/k) for spring; T = 2*pi*sqrt(L/g) for pendulum.
- Frequency: f = 1/T (Hz).

Energy in SHM:
- KE = (1/2)*m*omega^2*(A^2 - x^2) -- maximum at x=0.
- PE = (1/2)*k*x^2 = (1/2)*m*omega^2*x^2 -- maximum at x=+/-A.
- Total energy E = (1/2)*k*A^2 = constant (conserved in ideal SHM).

Examples: Mass on spring, simple pendulum (small angles <15 degrees), vibrating tuning fork.

Damped SHM: Friction reduces amplitude over time. Resonance occurs when driving frequency matches natural frequency omega_0 = sqrt(k/m).
        """
    },
    {
        "id": "doc_006", "topic": "Wave Optics and Interference",
        "text": """
Wave optics treats light as a wave, explaining interference, diffraction, and polarization.

Superposition Principle: When two waves meet, resultant displacement = algebraic sum of individual displacements.

Interference: Superposition of coherent waves produces bright (constructive) and dark (destructive) fringes.
- Constructive: path difference = n*lambda (n = 0, 1, 2,...) gives bright fringe.
- Destructive: path difference = (2n-1)*lambda/2 gives dark fringe.

Young's Double Slit Experiment (YDSE):
- Two slits separation d, screen distance D.
- Fringe width: beta = lambda*D/d
- nth bright fringe position: y_n = n*lambda*D/d
- nth dark fringe position: y_n = (2n-1)*lambda*D/(2d)

Diffraction: Bending of light around obstacles or slits.
- Single slit: First minimum at sin(theta) = lambda/a, where a = slit width.
- Diffraction grating: d*sin(theta) = n*lambda.

Polarization: Only transverse waves can be polarized. Unpolarized light has oscillations in all planes.
- Malus's Law: I = I0*cos^2(theta), where theta = angle between polarizer and analyzer.

Coherence: Sources must have same frequency and constant phase difference for sustained interference.

Applications: Anti-reflection coatings, holography, optical fibers, laser interferometers.
        """
    },
    {
        "id": "doc_007", "topic": "Magnetic Force and Faradays Law",
        "text": """
Magnetism and electromagnetic induction are foundational to generators, motors, and transformers.

Magnetic Force on Moving Charge: F = q*v x B (cross product).
Magnitude: F = q*v*B*sin(theta). Unit: Newton. This force is always perpendicular to both v and B.

Magnetic Force on Current-Carrying Wire: F = I*L x B. Magnitude: F = B*I*L*sin(theta).

Biot-Savart Law: Magnetic field due to small current element.
dB = (mu_0/4*pi)*(I*dl x r_hat)/r^2
For long straight wire: B = mu_0*I/(2*pi*r).
mu_0 = 4*pi*1e-7 T*m/A (permeability of free space).

Ampere's Law: closed_integral(B*dl) = mu_0*I_enclosed.

Faraday's Law of Electromagnetic Induction:
EMF = -d(Phi_B)/dt
Where Phi_B = B*A*cos(theta) = magnetic flux (Weber, Wb = T*m^2).
Negative sign = Lenz's Law: induced current opposes the change in flux that caused it.

Motional EMF: EMF = B*L*v for conductor of length L moving with velocity v perpendicular to field B.

Transformers: V1/V2 = N1/N2. Power conserved: V1*I1 = V2*I2.

Self-Inductance (L): EMF = -L*dI/dt. Unit: Henry (H). Energy stored: E = (1/2)*L*I^2.
        """
    },
    {
        "id": "doc_008", "topic": "Photoelectric Effect and Quantum Physics",
        "text": """
Quantum physics describes matter and energy at atomic scales where classical physics fails.

Photoelectric Effect: Electrons emitted from metal surface when light of sufficient frequency shines on it.
Key findings:
- Emission ONLY if f >= f0 (threshold frequency), regardless of intensity.
- Maximum KE of emitted electrons: KE_max = h*f - phi
Where h = 6.626e-34 J*s (Planck's constant), phi = h*f0 = work function of the metal.
- Intensity affects number of electrons, not their energy.
- Proved light behaves as photons (particle nature). Einstein won Nobel Prize 1921.

de Broglie Hypothesis (Wave-Particle Duality):
Every moving particle has associated wavelength: lambda = h/p = h/(m*v)
Where p = momentum.

Heisenberg's Uncertainty Principle:
delta_x * delta_p >= h/(4*pi)  and  delta_E * delta_t >= h/(4*pi)
Position and momentum cannot both be known exactly simultaneously.

Bohr's Model of Hydrogen Atom:
- Electrons occupy discrete shells (n = 1, 2, 3,...) without radiating.
- Energy of nth level: E_n = -13.6/n^2 eV (negative = bound state).
- Photon emitted when electron drops: E_photon = E_higher - E_lower = h*f.
- Radius of nth orbit: r_n = 0.529*n^2 Angstrom (Bohr radius a0 = 0.529 Angstrom).
- Ground state (n=1): E1 = -13.6 eV. Ionization energy = 13.6 eV.

1 eV = 1.6e-19 J.
        """
    },
    {
        "id": "doc_009", "topic": "Rotational Motion and Moment of Inertia",
        "text": """
Rotational mechanics extends Newton's Laws to rotating bodies using angular analogs.

Angular Quantities:
- Angular displacement: theta (radians). 2*pi rad = 360 degrees.
- Angular velocity: omega = d(theta)/dt (rad/s).
- Angular acceleration: alpha = d(omega)/dt (rad/s^2).
- Linear-angular relations: v = r*omega, a_t = r*alpha, a_c = r*omega^2 = v^2/r (centripetal).

Rotational Kinematic Equations:
- omega = omega_0 + alpha*t
- theta = omega_0*t + (1/2)*alpha*t^2
- omega^2 = omega_0^2 + 2*alpha*theta

Moment of Inertia (I): Resistance to rotational acceleration. I = sum(m_i * r_i^2). Unit: kg*m^2.
Standard values (uniform bodies):
- Solid cylinder/disk (central axis): I = (1/2)*M*R^2
- Hollow cylinder/ring (central axis): I = M*R^2
- Solid sphere (diameter): I = (2/5)*M*R^2
- Hollow sphere (diameter): I = (2/3)*M*R^2
- Thin rod (through center): I = (1/12)*M*L^2
- Thin rod (through end): I = (1/3)*M*L^2

Parallel Axis Theorem: I = I_cm + M*d^2

Torque: tau = r x F. Magnitude: tau = r*F*sin(theta). Newton's 2nd law for rotation: tau_net = I*alpha.
Rotational KE: KE_rot = (1/2)*I*omega^2.
Angular Momentum: L = I*omega. Unit: kg*m^2/s.
Conservation of L: If tau_net = 0, then L = constant.
        """
    },
    {
        "id": "doc_010", "topic": "Gravitation and Keplers Laws",
        "text": """
Gravitation is the attractive force between all masses, governing celestial motion.

Newton's Law of Universal Gravitation:
F = G*m1*m2/r^2
Where G = 6.674e-11 N*m^2/kg^2, m1, m2 = masses (kg), r = distance between centers (m).

Acceleration due to gravity at Earth's surface: g = G*M_E/R_E^2 approx 9.8 m/s^2.
Variation with altitude h: g' = g*R^2/(R+h)^2 (decreases with altitude).
Variation with depth d: g_d = g*(1 - d/R).

Gravitational Potential Energy: PE = -G*m1*m2/r (zero at infinity; negative = bound).
Near Earth's surface: delta_PE = m*g*h (for small h).

Escape Velocity: Minimum speed to escape a planet's gravity:
v_esc = sqrt(2*G*M/R) = sqrt(2*g*R) approx 11.2 km/s for Earth.

Orbital Velocity (circular orbit at radius r):
v_orb = sqrt(G*M/r). At Earth's surface approx 7.9 km/s.

Kepler's Laws of Planetary Motion:
1. Law of Orbits: Planets move in elliptical orbits with the Sun at one focus.
2. Law of Areas: Line from planet to Sun sweeps equal areas in equal time intervals.
3. Law of Periods: T^2 is proportional to a^3, or T^2 = (4*pi^2/G*M)*a^3, where a = semi-major axis.

Geostationary Orbit: h approx 36,000 km, T = 24 hours.
        """
    },
    {
        "id": "doc_011", "topic": "Nuclear Physics and Radioactivity",
        "text": """
Nuclear physics studies atomic nuclei, their composition, stability, and decay.

Nuclear Composition: Protons (charge +e) + Neutrons (neutral) = Nucleons.
Atomic number Z = protons. Mass number A = Z + N (N = neutrons).
Isotopes: same Z (element), different N (mass).

Binding Energy: Energy to completely separate all nucleons.
BE = delta_m * c^2, where delta_m = mass defect = (Z*m_p + N*m_n) - M_nucleus.
c = 3e8 m/s. Binding energy per nucleon peaks at iron-56 (most stable nucleus).

Types of Radioactive Decay:
- Alpha (alpha): emits He-4 nucleus. A decreases by 4, Z decreases by 2.
- Beta-minus (beta-): n -> p + e- + antineutrino. Z increases by 1, A unchanged.
- Beta-plus (beta+): p -> n + e+ + neutrino. Z decreases by 1, A unchanged.
- Gamma (gamma): high-energy photon emission. A and Z unchanged.

Radioactive Decay Law:
N(t) = N0 * e^(-lambda*t)
Where lambda = decay constant, N0 = initial number of nuclei.

Half-life: T_half = 0.693/lambda = time for half the nuclei to decay.
After n half-lives: N = N0 * (1/2)^n.

Activity: A = lambda*N. Unit: Becquerel (Bq) = 1 decay/s. 1 Curie (Ci) = 3.7e10 Bq.

Nuclear Fission: Heavy nucleus (e.g., U-235) splits releasing ~200 MeV per event. Used in reactors.
Nuclear Fusion: Light nuclei combine (e.g., H + H -> He), releasing even more energy. Powers the Sun.
Mass-Energy Equivalence: E = m*c^2.
        """
    },
    {
        "id": "doc_012", "topic": "Fluid Mechanics and Bernoullis Principle",
        "text": """
Fluid mechanics studies liquids and gases at rest (statics) and in motion (dynamics).

Pressure: P = F/A. Unit: Pascal (Pa = N/m^2). 1 atm = 101325 Pa = 760 mmHg.

Hydrostatic Pressure: Pressure at depth h: P = P0 + rho*g*h.
Where rho = fluid density (kg/m^3), g = 9.8 m/s^2.

Pascal's Law: Pressure applied to a confined fluid is transmitted equally in all directions.
Application: Hydraulic lift -- F1/A1 = F2/A2.

Archimedes' Principle: Buoyant force = weight of fluid displaced.
F_buoy = rho_fluid * V_submerged * g.
Object floats if rho_object < rho_fluid.

Continuity Equation (Mass conservation for steady flow):
A1*v1 = A2*v2
Where A = cross-sectional area (m^2), v = fluid velocity (m/s). Fluid speeds up when pipe narrows.

Bernoulli's Principle (Energy conservation for ideal fluid):
P + (1/2)*rho*v^2 + rho*g*h = constant along a streamline.
Implications: Higher velocity means lower pressure (and vice versa) at same height.

Applications of Bernoulli:
- Airplane lift: faster airflow over curved top wing gives lower pressure above, net upward lift.
- Venturi meter: flow rate measurement using pressure drop.
- Spray atomizer, carburetor, Pitot tube (airspeed measurement).

Viscosity (eta): Internal friction in fluids. Unit: Pa*s. Water approx 1e-3 Pa*s.
Stokes' Law: Drag on sphere = 6*pi*eta*r*v.
Surface Tension (T): T = F/L. Capillary rise: h = 2*T*cos(theta)/(rho*g*r).
        """
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE MODELS AND VECTORSTORE (called once; cached by Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def init_resources():
    """Initialise LLM, embedder, and ChromaDB. Returns (llm, embedder, collection)."""
    llm = ChatGroq(model=MODEL_NAME, temperature=0.0, api_key=GROQ_API_KEY)

    # SentenceTransformer — real semantic embeddings (downloads ~90 MB on first run)
    embedder = SentenceTransformer(EMBED_MODEL)

    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.create_collection(
            name="physics_kb",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        chroma_client.delete_collection("physics_kb")
        collection = chroma_client.create_collection(
            name="physics_kb",
            metadata={"hnsw:space": "cosine"},
        )

    docs_text:  List[str]         = [doc["text"]  for doc in KNOWLEDGE_BASE]
    docs_ids:   List[str]         = [doc["id"]    for doc in KNOWLEDGE_BASE]
    docs_meta:  List[Metadata]    = [{"topic": doc["topic"]} for doc in KNOWLEDGE_BASE]
    embeddings: List[List[float]] = embedder.encode(docs_text).tolist()
    collection.add(
        documents=docs_text,
        embeddings=embeddings,
        ids=docs_ids,
        metadatas=docs_meta,
    )

    return llm, embedder, collection

# ─────────────────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_memory_node():
    def memory_node(state: CapstoneState) -> dict:
        messages     = list(state.get("messages", []))
        question     = state["question"]
        student_name = state.get("student_name")
        messages.append({"role": "user", "content": question})
        messages = messages[-MAX_WINDOW:]
        name_match = re.search(r"my name is\s+([A-Za-z]+)", question, re.IGNORECASE)
        if name_match:
            student_name = name_match.group(1).capitalize()
        return {
            "messages":     messages,
            "student_name": student_name,
            "eval_retries": state.get("eval_retries", 0),
        }
    return memory_node


ROUTER_PROMPT = """You are a router for a B.Tech Physics Study Buddy assistant.

Based on the student question, choose exactly ONE route:
- retrieve   : question is about a physics concept, law, formula, theory, or derivation from the syllabus.
- tool       : question needs current date/time OR requires numeric arithmetic/calculation only.
- memory_only: greeting, simple follow-up, or question answerable purely from chat history.

Reply with exactly ONE word: retrieve OR tool OR memory_only

Student question: {question}
"""

def build_router_node(llm):
    def router_node(state: CapstoneState) -> dict:
        prompt   = ROUTER_PROMPT.format(question=state["question"])
        response = llm.invoke(prompt)
        route    = response.content.strip().lower().split()[0]
        if route not in ("retrieve", "tool", "memory_only"):
            route = "retrieve"
        print(f"  [router] route={route}")
        return {"route": route}
    return router_node


def build_retrieval_node(embedder, collection):
    def retrieval_node(state: CapstoneState) -> dict:
        question = state["question"]
        q_emb    = embedder.encode([question]).tolist()   # List[List[float]]
        results  = collection.query(query_embeddings=q_emb, n_results=TOP_K)
        chunks   = results["documents"][0]
        metas    = results["metadatas"][0]
        sources  = [m["topic"] for m in metas]
        context  = "\n\n".join(
            f"[{topic}]\n{chunk.strip()}" for topic, chunk in zip(sources, chunks)
        )
        print(f"  [retrieval] sources={sources}")
        return {"retrieved": context, "sources": sources}
    return retrieval_node


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


def build_tool_node(llm):
    def get_datetime_info() -> str:
        now = datetime.datetime.now()
        return (f"Current date: {now.strftime('%A, %d %B %Y')}\n"
                f"Current time: {now.strftime('%I:%M %p')}")

    def safe_calculator(question: str) -> str:
        extract_prompt = (
            f"Extract ONLY the mathematical expression from this physics question to evaluate.\n"
            f"Reply with ONLY a Python-evaluable expression (e.g., '50/9.8'). "
            f"If no numeric calculation needed, reply: NONE\n\nQuestion: {question}"
        )
        try:
            resp = llm.invoke(extract_prompt)
            expr = resp.content.strip()
            if expr.upper() == "NONE" or not expr:
                return "No numeric calculation could be extracted."
            safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            safe_dict["__builtins__"] = {}
            result = eval(expr, safe_dict)  # noqa: S307
            return f"Calculation: {expr} = {result:.6g}"
        except Exception as exc:
            return f"Calculator error: {exc}. Please check the expression."

    def tool_node(state: CapstoneState) -> dict:
        q = state["question"].lower()
        if any(kw in q for kw in ("time", "date", "today", "day", "when is")):
            tool_result = get_datetime_info()
        else:
            tool_result = safe_calculator(state["question"])
        print(f"  [tool] {tool_result[:80]}")
        return {"tool_result": tool_result, "retrieved": "", "sources": []}

    return tool_node


ANSWER_SYSTEM = """You are PhysicsBot -- an expert B.Tech Physics Study Buddy assistant.

STRICT RULES:
1. If RETRIEVED CONTEXT is provided, base your answer ONLY on it. Never invent formulas, constants, or claims not present in the context.
2. If TOOL RESULT is provided, use it directly to answer.
3. If neither is provided, answer from CONVERSATION HISTORY only.
4. If you cannot answer from available information, say clearly:
   "I don't have information on that topic in my knowledge base. Please refer to your textbook or ask your professor. You can also reach us at {support_email}."
5. NEVER fabricate numerical values, equations, or physics facts.
6. Format answers clearly: explain each variable in formulas.
7. Be encouraging, student-friendly. Use student's name if known.
8. {{retry_instruction}}
""".format(support_email=SUPPORT_EMAIL)


def build_answer_node(llm):
    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        student_name = state.get("student_name", "")
        eval_retries = state.get("eval_retries", 0)

        retry_instruction = ""
        if eval_retries > 0:
            retry_instruction = (
                "IMPORTANT: Previous answer was flagged for low faithfulness. "
                "Be strictly conservative -- only include statements directly supported by context."
            )

        system = ANSWER_SYSTEM.format(retry_instruction=retry_instruction)
        if student_name:
            system += f"\nStudent's name: {student_name}"

        parts = []
        if retrieved:
            parts.append(f"RETRIEVED CONTEXT:\n{retrieved}")
        if tool_result:
            parts.append(f"TOOL RESULT:\n{tool_result}")
        if len(messages) > 1:
            history = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages[-4:-1]
            )
            parts.append(f"CONVERSATION HISTORY:\n{history}")
        parts.append(f"STUDENT QUESTION: {question}")
        parts.append("Provide a clear, accurate, and complete answer:")

        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content="\n\n".join(parts))
        ])
        answer = response.content.strip()
        print(f"  [answer] len={len(answer)}, retries={eval_retries}")
        return {"answer": answer}
    return answer_node


EVAL_PROMPT = """You are a faithfulness evaluator for a Physics Study Buddy AI.

Rate how faithfully the ANSWER is grounded in the CONTEXT below.
0.0 = completely fabricated, no relation to context.
1.0 = every claim is directly supported by context.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."""


def build_eval_node(llm):
    def eval_node(state: CapstoneState) -> dict:
        retrieved    = state.get("retrieved", "")
        answer       = state.get("answer", "")
        eval_retries = state.get("eval_retries", 0)

        if not retrieved.strip():
            print("  [eval] No context -- faithfulness=1.0 (skipped)")
            return {"faithfulness": 1.0, "eval_retries": eval_retries}

        prompt = EVAL_PROMPT.format(context=retrieved[:2000], answer=answer)
        try:
            resp         = llm.invoke(prompt)
            faithfulness = float(resp.content.strip().split()[0])
            faithfulness = max(0.0, min(1.0, faithfulness))
        except Exception:
            faithfulness = 0.5

        gate = "PASS" if faithfulness >= FAITH_THRESHOLD else "RETRY"
        print(f"  [eval] faithfulness={faithfulness:.2f} -> {gate} (retries={eval_retries})")
        if gate == "RETRY":
            eval_retries += 1
        return {"faithfulness": faithfulness, "eval_retries": eval_retries}
    return eval_node


def save_node(state: CapstoneState) -> dict:
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": state.get("answer", "")})
    print(f"  [save] Total messages: {len(messages)}")
    return {"messages": messages, "tool_result": "", "eval_retries": 0}

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(llm, embedder, collection):
    """Build and compile the LangGraph StateGraph."""

    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        return r if r in ("retrieve", "tool", "memory_only") else "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        if (state.get("faithfulness", 1.0) < FAITH_THRESHOLD and
                state.get("eval_retries", 0) < MAX_EVAL_RETRIES):
            return "answer"
        return "save"

    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   build_memory_node())
    graph.add_node("router",   build_router_node(llm))
    graph.add_node("retrieve", build_retrieval_node(embedder, collection))
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     build_tool_node(llm))
    graph.add_node("answer",   build_answer_node(llm))
    graph.add_node("eval",     build_eval_node(llm))
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    graph.add_conditional_edges("router", route_decision, {
        "retrieve":    "retrieve",
        "tool":        "tool",
        "memory_only": "skip",
    })
    graph.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",
        "save":   "save",
    })

    app = graph.compile(checkpointer=MemorySaver())
    print("Graph compiled successfully")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state(question: str) -> CapstoneState:
    """Return a fully-populated CapstoneState dict for a new question."""
    return {
        "question":     question,
        "messages":     [],
        "route":        "",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 1.0,
        "eval_retries": 0,
        "student_name": None,
        "topic_asked":  None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL ask() -- used by capstone_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_app():
    """Lazy-initialise and cache the compiled graph (once per process)."""
    llm, embedder, collection = init_resources()
    return build_graph(llm, embedder, collection)


def ask(question: str, thread_id: str = "default", messages=None, student_name=None) -> dict:
    """Public API used by Streamlit. Returns full result dict."""
    app    = _get_app()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    initial = make_initial_state(question)
    if messages:
        initial["messages"] = list(messages)
    if student_name:
        initial["student_name"] = student_name
    return app.invoke(initial, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising resources...")
    llm, embedder, collection = init_resources()
    app = build_graph(llm, embedder, collection)

    def cli_ask(question, thread_id="cli-session"):
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(make_initial_state(question), config=config)
        return result

    print("\nPhysicsBot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        r = cli_ask(q)
        print(f"\nPhysicsBot: {r['answer']}\n")