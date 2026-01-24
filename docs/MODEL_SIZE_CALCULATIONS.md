\section{Knowledge Distillation Constraints for Ensemble Models}
\label{sec:kd_constraints}

% Previous sections have established the hard parameter budget available for onboard models
% (175K--420K parameters depending on precision and CAN scheduling). 
% This section derives the knowledge distillation framework required to train effective student
% ensembles within these constraints, establishing principled teacher sizing and multi-model 
% ensemble decomposition based on empirical distillation scaling laws.

\subsection{Problem Formulation: Ensemble Parameter Allocation}

The cascading knowledge distillation framework must solve the following optimization problem:

\begin{equation}
\text{Maximize} \quad \text{Student Ensemble Performance}
\end{equation}

\noindent subject to:

\begin{align}
\text{Subject to:} \quad 
&\sum_{i=1}^{M} N_{s,i} \leq N_{\text{onboard, max}}  \label{eq:memory_constraint}\\
&N_{s,i} \leq \frac{\text{FLOPs}_{\text{available}}}{2.0} \quad \forall i \in \{1, \ldots, M\} \label{eq:latency_constraint}\\
&\frac{N_{t,j}}{N_{s,i}} \in [5, 50] \quad \text{(compression ratio, safe range)} \label{eq:compression_constraint}\\
&\text{All models distilled from shared teacher ensemble} \label{eq:ensemble_structure}
\end{align}

where:
\begin{itemize}
\item $M$ = number of student models in the ensemble
\item $N_{s,i}$ = parameters in student model $i$
\item $N_{t,j}$ = parameters in teacher model $j$
\item $N_{\text{onboard, max}}$ = total parameter budget for deployment (\(\approx 175\text{--}420\text{ K}\) from \S\ref{sec:latency})
\item $\text{FLOPs}_{\text{available}}$ = computational budget from CAN cycle (350 K FLOPs in 7 ms)
\end{itemize}

\subsection{Capacity Gap and the Teacher Assistant Problem}

A fundamental constraint in knowledge distillation is the \textbf{capacity gap problem}: empirical research demonstrates that traditional KD becomes ineffective when the teacher-student parameter ratio exceeds approximately 10:1~\cite{Gap-KD2025,Mirzadeh-TAKD2020}.

\begin{theorem}[Capacity Gap Threshold~\cite{Gap-KD2025}]
For a student model with $N_s$ parameters learning from a teacher with $N_t$ parameters, KD performance improvement over supervised self-training is:

\begin{equation}
\Delta \text{Performance}(r) = 
\begin{cases}
\text{Positive} & \text{if } r = \frac{N_t}{N_s} \leq 10 \\
\text{Negative or marginal} & \text{if } r > 10
\end{cases}
\end{equation}

When $r > 10$, student models fail to effectively learn the teacher's intra-class decision boundaries due to distributional mismatch~\cite{A-Good-Teacher-2025}.
\end{theorem}

\noindent This motivates the \textbf{Teacher Assistant} (TA) approach: introduce an intermediate model $N_{ta}$ such that:

\begin{equation}
\frac{N_t}{N_{ta}} \leq 10 \quad \text{and} \quad \frac{N_{ta}}{N_s} \leq 10
\end{equation}

\noindent Two-stage distillation then flows: Teacher $\rightarrow$ TA $\rightarrow$ Student.

\subsection{Safe Compression Ratios by Task Complexity}

Empirical distillation research identifies task-dependent compression thresholds~\cite{Gap-KD2025,Mirzadeh-TAKD2020}:

\begin{equation}
N_{\text{teacher}} = \kappa \cdot N_{\text{student}}
\label{eq:compression_ratio}
\end{equation}

where $\kappa$ (compression ratio) depends on task complexity:

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
\textbf{Task Type} & \textbf{Decision Complexity} & \boldsymbol{\kappa} & \textbf{Architecture} \\
\midrule
Binary classification & Very simple & 50--100× & No TA required \\
Anomaly detection & Moderate & 20--30× & TA recommended if $\kappa > 10$ \\
Multi-class recognition & Complex & 5--10× & Standard KD \\
Safety-critical & Highest assurance & 3--5× & Conservative with TA \\
\bottomrule
\end{tabular}
\label{tab:compression_ratios}
\end{table}

\noindent For the automotive CAN-based anomaly detection task (binary: attack vs.\ benign with multi-class fine-graining), the appropriate range is:

\begin{equation}
\kappa \in [20, 30] \quad \text{(use TA if } \kappa > 10\text{)}
\label{eq:automotive_compression}
\end{equation}

\subsection{Distillation Scaling Laws: Optimal Compute Allocation}

Recent empirical research~\cite{Busbridge-DistillationScaling2025} establishes distillation scaling laws that predict student performance given a total training compute budget $C_{\text{total}}$ and its allocation between teacher and student training:

\begin{equation}
L_S(N_s, D_s, N_t, D_t) \approx A_s (N_s)^{-\alpha_s} + \text{KL}(p_t || p_s) 
\label{eq:distillation_loss}
\end{equation}

where:
\begin{itemize}
\item $L_S$ = student cross-entropy loss
\item $N_s, N_t$ = student and teacher parameter counts
\item $D_s, D_t$ = training data (tokens or examples) for student and teacher respectively
\item $\alpha_s$ = scaling exponent (empirically $\approx 0.07$ for language models)
\item $\text{KL}(p_t || p_s)$ = KL divergence between teacher and student output distributions
\end{itemize}

\noindent The critical finding is that optimal allocation of $C_{\text{total}}$ depends on whether the teacher \textbf{already exists} or \textbf{must be trained}:

\begin{theorem}[Optimal Distillation Scenarios~\cite{Busbridge-DistillationScaling2025}]
\mbox{}
\begin{enumerate}
\item \textbf{Multiple students, existing teacher}: Distillation outperforms supervised learning. Teacher training cost is amortized across all $M$ students.

\item \textbf{Multiple students, train teacher}: Distillation still outperforms. Optimal allocation increases teacher compute as total budget grows (to produce better soft targets).

\item \textbf{Single student, train teacher}: Supervised learning is often preferable. Compute better spent on purely training one larger student than splitting between teacher and student.
\end{enumerate}
\end{theorem}

\noindent \textbf{Application to automotive ensemble}: Since we have $M \geq 2$ student models and the teacher is trained offline, Scenario 1 or 2 applies. Distillation is advantageous.

\subsubsection{Teacher Size Scaling with Student Ensemble}

For the ensemble distillation scenario (training one teacher ensemble to distill $M$ independent student models), empirical data reveal a \textbf{linear scaling relationship}~\cite{Towards-Law-of-Capacity-Gap2025}:

\begin{equation}
N_{\text{teacher, optimal}} \approx \beta_0 + \beta_1 \cdot N_{\text{student, avg}}
\label{eq:teacher_scaling_linear}
\end{equation}

where:
\begin{itemize}
\item $\beta_0 \approx 50\text{K}$ (base teacher size for any task)
\item $\beta_1 \approx 20\text{--}30$ (compression ratio, task-dependent)
\item $N_{\text{student, avg}} = \frac{1}{M} \sum_{i=1}^M N_{s,i}$ (average student parameters)
\end{itemize}

This linear relationship enables straightforward sizing: given a desired average student size, multiply by the task-specific $\beta_1$ to determine teacher parameters.

\subsection{Multi-Model Ensemble Decomposition}

\subsubsection{Ensemble Architecture}

The cascading knowledge distillation framework decomposes into:

\begin{equation}
\text{Ensemble} = \{\text{Teacher Ensemble}\} \rightarrow \{\text{TA (optional)}\} \rightarrow \{\text{M Student Models}\}
\label{eq:ensemble_architecture}
\end{equation}

\noindent Detailing each component:

\paragraph{Teacher Ensemble (Offline)}

Consisting of $K$ independent teacher models, each with $N_{t,k}$ parameters:

\begin{equation}
\sum_{k=1}^{K} N_{t,k} = K \cdot N_{t, \text{avg}} \quad \text{(total teacher parameters, offline)}
\end{equation}

\noindent Typical $K \in [2, 5]$ for sufficient diversity and ensemble gain~\cite{Online-Ensemble-Compression2020}. Multiple teachers provide complementary representations of the input space.

\paragraph{Teacher Assistant (Conditional)}

If maximum compression ratio from teacher to student exceeds 10:1, introduce intermediate TA models:

\begin{equation}
N_{ta,j} = \frac{N_{t,j} + N_{s,i}}{2} \quad \text{(geometric mean, bridging gap)}
\label{eq:ta_sizing}
\end{equation}

Two-stage distillation:
\begin{enumerate}
\item \textbf{Stage 1}: Teacher ensemble $\rightarrow$ TA models (distillation)
\item \textbf{Stage 2}: TA ensemble $\rightarrow$ Student ensemble (distillation)
\end{enumerate}

\paragraph{Student Ensemble (Onboard)}

Consisting of $M$ independent student models for heterogeneous inference:

\begin{equation}
\sum_{i=1}^{M} N_{s,i} \leq N_{\text{onboard, max}}
\end{equation}

\noindent Architectural diversity is critical~\cite{Online-Ensemble-Compression2020}: each student should have distinct layer configurations, connectivity patterns, or activation functions to learn complementary feature representations.

\subsubsection{Student Ensemble Member Sizing}

For $M$ equally-sized student models (simplest case), each member has:

\begin{equation}
N_{s,i} = \frac{N_{\text{onboard, max}}}{M} \quad \forall i \in \{1, \ldots, M\}
\label{eq:uniform_student_sizing}
\end{equation}

More generally, student models can be heterogeneous, provided the total budget is respected. Asymmetric sizing allows:
\begin{itemize}
\item Specialized architectures (e.g., one attention-based GAT, one LSTM-based, one CNN-based)
\item Varying precision (one FP32, one INT8 for redundancy)
\item Hierarchical complexity (one ``expert'' model, others ``generalists'')
\end{itemize}

\subsection{Parameter Budget Derivation for Ensemble}

\subsubsection{Step 1: Onboard Constraint}

From \S\ref{sec:latency}, the hard constraint is:

\begin{equation}
N_{\text{onboard, max}} = \min(N_{\text{memory}}, N_{\text{latency}})
\end{equation}

For the automotive scenario (FP32, CAN-hard constraint):

\begin{equation}
N_{\text{onboard, max}} \approx 175\text{ K parameters}
\end{equation}

Or with INT8 quantization:

\begin{equation}
N_{\text{onboard, max}} \approx 350\text{--}420\text{ K parameters}
\end{equation}

\subsubsection{Step 2: Distribute Across $M$ Students}

Choose number of students $M$ based on desired diversity and cost-accuracy tradeoff. Empirical evidence suggests $M \in [2, 5]$ for near-optimal results~\cite{Online-Ensemble-Compression2020, The-Cost-of-Ensembling2025}:

\begin{equation}
M = 3 \quad \text{(typical choice: balance accuracy gains vs.\ inference cost)}
\label{eq:ensemble_size}
\end{equation}

Each student then has budget:

\begin{equation}
N_{s,\text{per model}} = \frac{N_{\text{onboard, max}}}{M} = \frac{175\text{ K}}{3} \approx 58.3\text{ K parameters (FP32)}
\label{eq:student_per_model}
\end{equation}

\noindent or, with INT8:

\begin{equation}
N_{s,\text{per model}} = \frac{350\text{ K}}{3} \approx 116.7\text{ K parameters (INT8)}
\label{eq:student_per_model_int8}
\end{equation}

\subsubsection{Step 3: Determine Teacher Size}

Apply the compression ratio from \eqref{eq:automotive_compression}:

\begin{equation}
N_{t,\text{per model}} = \kappa \cdot N_{s,\text{per model}}
\label{eq:teacher_per_model}
\end{equation}

\noindent For $\kappa = 20$ (moderate compression) and $M = 3$ students:

\begin{equation}
N_{t,\text{per model}} = 20 \times 58.3\text{ K} = 1.166\text{ M parameters (FP32)}
\label{eq:teacher_example_fp32}
\end{equation}

\noindent or, with INT8 on students:

\begin{equation}
N_{t,\text{per model}} = 20 \times 116.7\text{ K} = 2.334\text{ M parameters}
\label{eq:teacher_example_int8}
\end{equation}

If using $\kappa = 30$ (more aggressive compression, with intermediate TA):

\begin{equation}
N_{t,\text{per model}} = 30 \times 58.3\text{ K} = 1.749\text{ M parameters}
\end{equation}

\subsubsection{Step 4: Optional Teacher Assistant Sizing}

If compression ratio $\kappa > 10$, introduce intermediate TA models to avoid capacity gap failure:

\begin{equation}
N_{ta} = \sqrt{N_t \cdot N_s} \quad \text{(geometric mean)}
\label{eq:ta_geometric_mean}
\end{equation}

For the example with $\kappa = 20$:
\begin{itemize}
\item $N_t = 1.166$ M
\item $N_s = 58.3$ K
\item $N_{ta} = \sqrt{1.166 \times 10^6 \times 58.3 \times 10^3} \approx 260$ K parameters
\end{itemize}

Verify both stage ratios:
\begin{align}
\frac{N_t}{N_{ta}} &= \frac{1.166 \text{ M}}{260 \text{ K}} \approx 4.5 \quad \checkmark \text{(< 10)} \\
\frac{N_{ta}}{N_s} &= \frac{260 \text{ K}}{58.3 \text{ K}} \approx 4.5 \quad \checkmark \text{(< 10)}
\end{align}

\noindent Both stages have safe compression ratios.

\subsection{Complete Parameter Sizing Example}

\subsubsection{Scenario: FP32 Precision, $M=3$ Students, $\kappa=20$ Compression}

\begin{table}[H]
\centering
\begin{tabular}{lllll}
\toprule
\textbf{Component} & \textbf{Count} & \textbf{Params/Model} & \textbf{Total Params} & \textbf{Constraint} \\
\midrule
\textbf{Student (Onboard)} & 3 & 58.3 K & 175 K & Memory + Latency \\
\textbf{Teacher (Offline)} & 3 & 1.166 M & 3.498 M & None (offline) \\
\textbf{TA (Optional, Offline)} & 3 & 260 K & 780 K & None (offline) \\
\midrule
\textbf{Compression Ratios} & & & & \\
\quad Teacher:TA & & & 1.166 M : 260 K & $\approx 4.5:1$ \\
\quad TA:Student & & & 260 K : 58.3 K & $\approx 4.5:1$ \\
\quad Teacher:Student & & & 1.166 M : 58.3 K & $= 20:1$ \\
\bottomrule
\end{tabular}
\label{tab:ensemble_sizing_example_fp32}
\end{table}

\subsubsection{Scenario: INT8 Precision, $M=3$ Students, $\kappa=20$ Compression}

\begin{table}[H]
\centering
\begin{tabular}{lllll}
\toprule
\textbf{Component} & \textbf{Count} & \textbf{Params/Model} & \textbf{Total Params} & \textbf{Note} \\
\midrule
\textbf{Student (Onboard)} & 3 & 116.7 K & 350 K & INT8: 4× speedup \\
\textbf{Teacher (Offline)} & 3 & 2.334 M & 7.002 M & FP32 precision \\
\textbf{TA (Optional, Offline)} & 3 & 520 K & 1.560 M & FP32 precision \\
\midrule
\textbf{Compression Ratios} & & & & \\
\quad Teacher:TA & & & 2.334 M : 520 K & $\approx 4.5:1$ \\
\quad TA:Student & & & 520 K : 116.7 K & $\approx 4.5:1$ \\
\quad Teacher:Student & & & 2.334 M : 116.7 K & $= 20:1$ \\
\bottomrule
\end{tabular}
\label{tab:ensemble_sizing_example_int8}
\end{table}

\subsection{Ensemble Inference Cost and Accuracy Trade-offs}

\subsubsection{Cumulative Inference Cost}

Unlike training (where teacher cost is amortized across students), during inference all ensemble members run independently. Total inference FLOPs:

\begin{equation}
\text{FLOPs}_{\text{inference, ensemble}} = M \times \text{FLOPs}_{\text{per model}}
\label{eq:ensemble_inference_flops}
\end{equation}

With $M=3$ students and 58.3 K parameters each (assuming 2 FLOPs/parameter):

\begin{equation}
\text{FLOPs}_{\text{inference, ensemble}} = 3 \times (58.3 \text{ K} \times 2) = 349.8 \text{ K FLOPs}
\end{equation}

This equals the hard CAN latency budget, confirming three equally-sized students saturate the 7 ms window.

\subsubsection{Accuracy-Cost Trade-off: Ensemble Size}

Empirical research on ensemble models~\cite{The-Cost-of-Ensembling2025} reveals diminishing returns:

\begin{table}[H]
\centering
\begin{tabular}{lllll}
\toprule
\textbf{\# Models} & \textbf{Ensemble Accuracy Gain} & \textbf{Inference Cost} & \textbf{Cost per \% Gain} & \textbf{Recommendation} \\
\midrule
1 & Baseline & $1\times$ & --- & Single model, limited diversity \\
2 & $\sim 2$--3\% & $2\times$ & $0.7$--$1.5$ & Good balance \\
3 & $\sim 3$--4\% & $3\times$ & $0.8$--$1.3$ & \textbf{Optimal for most tasks} \\
4 & $\sim 4$--5\% & $4\times$ & $0.8$--$1.25$ & Diminishing gains \\
5+ & $\sim 5$--6\% & $5\times+$ & $\geq 0.8$ & Marginal, increased complexity \\
\bottomrule
\end{tabular}
\label{tab:ensemble_accuracy_cost}
\end{table}

\noindent Based on this, $M=3$ is the recommended choice for the automotive scenario.

\subsubsection{Accuracy-Driven vs.\ Efficiency-Driven Ensembles}

Two strategies for combining ensemble members~\cite{The-Cost-of-Ensembling2025}:

\begin{enumerate}
\item \textbf{Accuracy-Driven}: Include the highest-performing models, regardless of cost. Maximizes accuracy but increases inference latency.

\item \textbf{Efficiency-Driven}: Combine low-cost models that collectively maintain competitive accuracy. Lower latency footprint.
\end{enumerate}

\noindent For safety-critical automotive systems, a \textbf{hybrid approach} is recommended:
\begin{itemize}
\item Include one high-accuracy ``expert'' model (more parameters, careful training)
\item Include two efficient ``generalist'' models (smaller, faster)
\item Ensemble majority voting or weighted averaging based on per-model confidence
\end{itemize}

This balances safety (expert model acts as verification) with efficiency (generalists provide fast redundancy).

\subsection{Distillation Training Strategy}

\subsubsection{Loss Function Formulation}

For ensemble distillation with optional TA, the training loss combines three components:

\begin{equation}
L_{\text{total}} = \lambda_{\text{ce}} L_{\text{CE}} + \lambda_{\text{kd}} L_{\text{KD}} + \lambda_{\text{feat}} L_{\text{feat}}
\label{eq:total_loss}
\end{equation}

where:

\begin{align}
L_{\text{CE}} &= -\sum_c y_c \log(p_{s}(c)) \quad \text{(cross-entropy on hard labels)}\\
L_{\text{KD}} &= \text{KL}(p_t || p_s) = \sum_c p_t(c) \log\left(\frac{p_t(c)}{p_s(c)}\right) \quad \text{(knowledge distillation)}\\
L_{\text{feat}} &= \sum_{\ell} \| f^{(t)}_\ell - A(f^{(s)}_\ell) \|_2^2 \quad \text{(feature-level matching)}
\end{align}

\noindent where:
\begin{itemize}
\item $p_t(c), p_s(c)$ = teacher and student output probabilities for class $c$
\item $f^{(t)}_\ell, f^{(s)}_\ell$ = teacher and student activations at layer $\ell$
\item $A(\cdot)$ = adaptation layer (e.g., $1 \times 1$ convolution) to match dimensions
\item $\lambda_{\text{ce}}, \lambda_{\text{kd}}, \lambda_{\text{feat}}$ = loss weights (typical: $0.3, 0.5, 0.2$)
\end{itemize}

\subsubsection{Temperature-Scaled Softmax}

The KD loss uses a temperature parameter to control the softness of soft targets:

\begin{equation}
p_t(c) = \frac{\exp(z_t(c) / T)}{\sum_{c'} \exp(z_t(c') / T)}
\label{eq:temperature_softmax}
\end{equation}

where $z_t(c)$ are the teacher's logits and $T$ is the temperature. Higher $T$ (e.g., $T=20$) produces softer probability distributions, enabling the student to learn from the full distributional structure of the teacher's predictions. Typical values: $T \in [5, 20]$.

\subsubsection{Ensemble Teacher Aggregation}

When distilling from a teacher ensemble (multiple independent teacher models), predictions are aggregated before KD loss computation:

\begin{equation}
p_{\text{ensemble}}(c) = \frac{1}{K} \sum_{k=1}^{K} p_{t,k}(c)
\label{eq:ensemble_aggregation}
\end{equation}

This ensemble averaging provides more stable soft targets than any single teacher, improving student generalization~\cite{Online-Ensemble-Compression2020}.

\subsection{Summary: Principled Ensemble Parameter Sizing}

\subsubsection{Design Decision Tree}

\begin{enumerate}
\item \textbf{Determine onboard parameter budget}:
   \begin{equation}
   N_{\text{onboard, max}} = \text{min}(N_{\text{memory}}, N_{\text{latency}}) 
   \approx 175\text{--}420\text{ K (FP32 or INT8)}
   \end{equation}

\item \textbf{Choose ensemble size $M$} (recommended $M=3$):
   \begin{equation}
   N_{s,\text{per model}} = \frac{N_{\text{onboard, max}}}{M}
   \end{equation}

\item \textbf{Select compression ratio $\kappa$} based on task (anomaly detection: $\kappa \in [20, 30]$):
   \begin{equation}
   N_{t,\text{per model}} = \kappa \cdot N_{s,\text{per model}}
   \end{equation}

\item \textbf{Assess capacity gap}: If $\kappa > 10$, add intermediate TA models:
   \begin{equation}
   N_{ta} = \sqrt{N_t \cdot N_s}
   \end{equation}

\item \textbf{Verify all stages $< 10:1$ ratio} and deploy with two-stage KD if TA used.

\item \textbf{Train with ensemble loss} (\eqref{eq:total_loss}) using temperature-scaled softmax (\eqref{eq:temperature_softmax}) and ensemble teacher aggregation (\eqref{eq:ensemble_aggregation}).
\end{enumerate}

\subsubsection{Tabular Quick Reference}

\begin{table}[H]
\centering
\begin{tabular}{llll}
\toprule
\textbf{Configuration} & \textbf{Onboard Total} & \textbf{Per Student} & \textbf{Per Teacher} \\
\midrule
FP32, $M=3$, $\kappa=20$ & 175 K & 58.3 K & 1.166 M \\
FP32, $M=3$, $\kappa=30$ & 175 K & 58.3 K & 1.749 M \\
INT8, $M=3$, $\kappa=20$ & 350 K & 116.7 K & 2.334 M \\
INT8, $M=3$, $\kappa=20$ (with TA) & 350 K & 116.7 K & 2.334 M (TA: 520 K) \\
\bottomrule
\end{tabular}
\label{tab:quick_reference}
\end{table}

\subsection{Key Empirical Findings and Recommendations}

\begin{enumerate}
\item \textbf{Capacity gap is real}: Compression ratios $> 10:1$ without intermediaries lead to student learning failure. Use teacher assistants for safety-critical applications.

\item \textbf{Distillation beats supervised learning for ensembles}: With multiple students, distillation from a shared teacher outperforms independent supervised training of each student.

\item \textbf{Three models is optimal for automotive}: $M = 2$--3 students balance ensemble accuracy gains ($\sim 3$--4\%) against cumulative inference cost (matching CAN budget).

\item \textbf{Scaling laws enable principled allocation}: Linear relationship between optimal teacher and student sizes removes guesswork from parameter sizing.

\item \textbf{Feature-level distillation improves transfer}: Adding intermediate feature matching loss ($L_{\text{feat}}$) significantly improves student generalization, especially across heterogeneous architectures.

\item \textbf{Teacher ensemble provides stable targets}: Averaging predictions from multiple independent teachers reduces noise in soft targets, improving student robustness.

\item \textbf{INT8 quantization extends budget}: With INT8 quantization, onboard parameters can increase 2--2.5×, or alternatively reduce teacher size requirements.
\end{enumerate}

% End of section
