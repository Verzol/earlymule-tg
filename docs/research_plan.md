# THESIS RESEARCH PLAN

## EarlyMule-TG: Early Mule-Account Detection through Multi-Scale Temporal Role Transitions in Digital Banking Transaction Graphs

**Sinh viên:** Giang Tuấn Minh  
**Phát hiện sớm tài khoản trung gian gian lận trong ngân hàng số dựa trên chuyển đổi vai trò đa tỷ lệ trong đồ thị giao dịch theo thời gian**

**Thời gian triển khai:** 4 tháng / 16 tuần  
**Tên phương pháp đề xuất:** **EarlyMule-TG**  
**Mục tiêu mở rộng:** Có thể phát triển thành bài báo nếu hoàn thành đầy đủ temporal evaluation, early-detection protocol, ablation, label-scarcity analysis và external validation.

---

# 1. Bối cảnh và động lực nghiên cứu

Tài khoản trung gian gian lận (*money mule / mule account*) là tài khoản được sử dụng để tiếp nhận, luân chuyển, phân tán hoặc rút tiền có nguồn gốc bất hợp pháp. Trong ngân hàng số, một tài khoản mule có thể có hồ sơ định danh và lịch sử sử dụng ban đầu tương đối bình thường, do đó các mô hình chỉ dựa trên thuộc tính tĩnh hoặc từng giao dịch riêng lẻ khó phát hiện sớm.

Dấu hiệu quan trọng thường xuất hiện khi xem tài khoản trong **bối cảnh mạng giao dịch và diễn biến theo thời gian**, ví dụ:

- nhận tiền từ nhiều nguồn trong khoảng thời gian ngắn;
- thay đổi nhanh từ vai trò người nhận sang người chuyển tiếp;
- giữ tiền trong thời gian rất ngắn;
- chuyển phần lớn số tiền vừa nhận sang một hoặc nhiều tài khoản khác;
- xuất hiện đột biến về số lượng đối tác giao dịch;
- tham gia các chuỗi nhiều bước, fan-in/fan-out, splitting hoặc layering;
- thay đổi mạnh vai trò giao dịch so với lịch sử trước đó.

Các hướng nghiên cứu gần đây đã đạt tiến bộ đáng kể về:

- graph fraud detection;
- spatial-temporal graph learning;
- semi-supervised AML;
- fraud detection trên dynamic graph streams;
- graph fraud camouflage;
- heterogeneous temporal fraud detection.

Tuy nhiên, phần lớn công trình vẫn tập trung vào **fraudulent transaction**, **generic fraudulent node**, hoặc **money-laundering account classification**. Khoảng trống phù hợp cho thesis này là:

> **Mô hình hóa trực tiếp sự chuyển đổi vai trò của một tài khoản theo thời gian — từ người nhận, người gom tiền, người chuyển tiếp đến người phân tán — và khai thác các chuyển đổi này để phát hiện mule account càng sớm càng tốt trước khi chuỗi luân chuyển tiền hoàn thành.**

Thesis đề xuất:

\[
\boxed{
\text{Temporal Transaction Graph}
+
\text{Multi-Scale Role Dynamics}
+
\text{Early Account-Level Detection}
}
\]

và đánh giá bằng **strict chronological / point-in-time protocol** nhằm tránh temporal leakage.

---

# 2. Mục tiêu nghiên cứu

## 2.1. Mục tiêu tổng quát

Xây dựng và đánh giá một phương pháp học trên đồ thị giao dịch theo thời gian nhằm phát hiện sớm tài khoản mule trong mạng giao dịch ngân hàng số, trong đó mô hình kết hợp biểu diễn đồ thị động với các đặc trưng chuyển đổi vai trò giao dịch ở nhiều thang thời gian.

## 2.2. Mục tiêu cụ thể

1. Xây dựng pipeline dữ liệu giao dịch theo thời gian với cơ chế point-in-time feature construction, không sử dụng thông tin tương lai.
2. Thiết lập các baseline từ tabular ML, static GNN đến temporal GNN.
3. Tái lập hoặc thích nghi các ý tưởng chính từ các công trình mạnh gần đây như STA-GT, MLaD² và FiFrauD.
4. Xây dựng bộ đặc trưng hành vi mule ở nhiều cửa sổ thời gian.
5. Mô hình hóa sự thay đổi vai trò:
   - receiver;
   - collector;
   - relay;
   - distributor.
6. Xây dựng mô hình **EarlyMule-TG** kết hợp:
   - temporal graph representation;
   - multi-scale behavioral representation;
   - role-transition representation.
7. Đánh giá trong điều kiện mất cân bằng lớp mạnh.
8. Đánh giá trong điều kiện ít nhãn:
   - 1%;
   - 5%;
   - 10%;
   - 25%;
   - 100% labels.
9. Đánh giá khả năng phát hiện sớm bằng:
   - Recall@K;
   - Time-to-Detect;
   - Detection Lead Time.
10. Thực hiện ablation để xác định đóng góp riêng của temporal graph, multi-scale behavior và role-transition module.
11. Nếu tiến độ cho phép, kiểm tra external validity trên dataset thứ hai hoặc thứ ba.

---

# 3. Câu hỏi nghiên cứu

## RQ1 — Temporal graph value

**Does temporal graph learning improve mule-account detection over tabular machine learning and static GNN baselines under strict chronological evaluation?**

Tức là:

> Việc khai thác thứ tự thời gian của các giao dịch có cải thiện đáng kể khả năng phát hiện mule account so với mô hình tabular và static graph hay không?

## RQ2 — Multi-scale role transition

**Do multi-scale account-role transitions provide additional predictive value beyond generic temporal graph representations?**

Tức là:

> Các chuyển đổi vai trò nhận–gom–chuyển tiếp–phân tán ở nhiều thang thời gian có cung cấp thông tin bổ sung so với temporal GNN thông thường hay không?

## RQ3 — Early detection

**Can the proposed model identify mule accounts earlier in the laundering sequence while maintaining an operationally useful alert precision?**

Tức là:

> Mô hình có thể phát hiện tài khoản mule trước khi chuỗi chuyển tiền hoàn thành mà vẫn duy trì precision đủ tốt hay không?

## RQ4 — Label scarcity

**How robust is the proposed method when only a small fraction of mule labels is available?**

## RQ5 — Generalization

**Do the learned temporal-role representations remain useful across different transaction-network datasets and laundering patterns?**

RQ5 là câu hỏi mở rộng, chỉ bắt buộc nếu D1 đã hoàn chỉnh.

---

# 4. Giả thuyết nghiên cứu

## H1

\[
Temporal\ GNN > Static\ GNN > Tabular
\]

về PR-AUC, Recall@K và mule-account recall trên temporal holdout.

## H2

\[
Temporal\ GNN + MultiScaleRole
>
Temporal\ GNN
\]

đặc biệt với các mule typology có:

- rapid pass-through;
- fan-in/fan-out;
- relay chains;
- bursty transaction behavior.

## H3

Role-transition representation giúp giảm:

\[
TimeToDetect
\]

và tăng:

\[
DetectionLeadTime.
\]

## H4

Trong điều kiện ít nhãn:

\[
EarlyMule\text{-}TG
\]

vẫn duy trì lợi ích so với supervised baselines thuần túy.

## H5 — mở rộng

Lợi ích của temporal-role modeling vẫn còn quan sát được trên external AML / illicit-account dataset dù mức cải thiện có thể giảm.

---

# 5. Phạm vi khóa luận được khóa cho 4 tháng

## 5.1. Phần bắt buộc

Sinh viên phải hoàn thành:

- D1 data pipeline;
- point-in-time graph construction;
- temporal split;
- leakage audit;
- LightGBM/XGBoost baseline;
- static GraphSAGE hoặc GAT baseline;
- TGN hoặc TGAT baseline;
- một recent top-tier graph-fraud baseline được thích nghi;
- multi-scale mule behavior encoder;
- role-transition representation;
- EarlyMule-TG;
- class-imbalance handling;
- early-detection evaluation;
- ablation;
- source code + README + config tái lập.

## 5.2. Phần mở rộng

Chỉ triển khai nếu phần bắt buộc đã ổn định:

- MLaD²-style self-supervised pretraining;
- FiFrauD-style unsupervised anomaly score;
- TF-GNN-style historical feedback;
- graph camouflage robustness;
- external validation D2/D3;
- explainability;
- calibration;
- streaming inference benchmark.

## 5.3. Không làm trong 4 tháng

Không đặt mục tiêu bắt buộc:

- xây full banking production system;
- federated learning đa ngân hàng;
- graph foundation model;
- LLM/agent cho điều tra AML;
- full heterogeneous KYC-device-IP graph nếu dataset không hỗ trợ;
- reinforcement-learning fraud response;
- adversarial graph defense đầy đủ;
- huấn luyện mô hình hàng tỷ tham số;
- tuyên bố triển khai thực tế tại ngân hàng nếu không có real banking data.

Scope cốt lõi phải giữ ở:

\[
\boxed{
Early\ Mule\ Detection
+
Temporal\ Graph
+
Multi\text{-}Scale\ Role\ Transition
}
\]

---

# 6. Ba bài nghiên cứu neo chính

> **Lưu ý:** ba paper dưới đây là **research anchors**. Chúng không hoàn toàn cùng bài toán mule-account detection. Khi so sánh định lượng, phải chạy các phiên bản baseline trên **cùng dataset, cùng split và cùng target**.

---

## P1. STA-GT — temporal transaction-fraud anchor, P0

**Yue Tian, Guanjun Liu.**  
**Spatial-Temporal-Aware Graph Transformer for Transaction Fraud Detection.**  
*IEEE Transactions on Industrial Informatics*, Vol. 20, No. 11, pp. 12659–12668, 2024.  
DOI: <https://doi.org/10.1109/TII.2024.3423447>

### Ý tưởng chính

STA-GT kết hợp:

\[
Spatial\ Graph\ Structure
+
Temporal\ Encoding
+
Transformer
\]

để học đồng thời:

- local graph information;
- long-range/global interactions;
- temporal dependencies.

### Vai trò trong thesis

Đây là **paper gần nhất về mặt kỹ thuật** với hướng EarlyMule-TG.

### Thesis kế thừa

- transaction graph;
- temporal encoding;
- global interaction learning.

### Thesis khác biệt

STA-GT tập trung vào transaction fraud detection.

EarlyMule-TG tập trung vào:

\[
Account\text{-}level
+
Early\ Mule\ Detection
+
Role\ Transition.
\]

### Yêu cầu baseline

Phải triển khai một **STA-GT-inspired baseline** hoặc reproduction gần paper nhất có thể trên D1.

Nếu reproduction nguyên gốc không khả thi do dataset/schema khác, phải:

- giữ temporal encoding;
- giữ graph + global attention principle;
- ghi rõ deviation;
- chạy trên cùng account-level target của thesis.

---

## P2. MLaD² — AML/account-level and label-scarcity anchor, P0

**Xuejiao Luo, Xiaohui Han, Wenbo Zuo, Xiaoming Wu, Wenyin Liu.**  
**MLaD²: A Semi-Supervised Money Laundering Detection Framework Based on Decoupling Training.**  
*IEEE Transactions on Information Forensics and Security*, Vol. 19, pp. 4518–4533, 2024.  
DOI: <https://doi.org/10.1109/TIFS.2024.3380262>

### Ý tưởng chính

MLaD² sử dụng:

- graph-based account representation;
- transaction relationship network;
- self-supervised contrastive pretraining;
- decoupled fine-tuning;
- small-label AML setting.

### Vai trò trong thesis

MLaD² là anchor cho:

- account-level AML;
- sparse labels;
- self-supervised graph representation.

### Thesis kế thừa

- graph-based account representation;
- label-scarcity evaluation;
- optional contrastive pretraining.

### Thesis khác biệt

MLaD² không đặt trọng tâm vào:

- continuous temporal graph;
- early detection;
- multi-scale role transitions.

### Yêu cầu

Không bắt buộc reproduce toàn bộ MLaD² ngay từ đầu.

Bắt buộc ít nhất phải có **label-scarcity experiment**:

\[
1\%,5\%,10\%,25\%,100\%
\]

để so sánh Proposed với các baseline.

Nếu thời gian cho phép, thêm MLaD²-style pretraining như extension.

---

## P3. FiFrauD — dynamic graph stream anchor, P1

**Samira Khodabandehlou, S. Alireza Hashemi G.**  
**FiFrauD: Unsupervised Financial Fraud Detection in Dynamic Graph Streams.**  
*ACM Transactions on Knowledge Discovery from Data*, Vol. 18, No. 5, 2024.  
DOI: <https://doi.org/10.1145/3641857>

### Ý tưởng chính

FiFrauD xem financial transactions như:

\[
Dynamic\ Graph\ Stream
\]

và tập trung vào:

- online fraud detection;
- unsupervised scoring;
- evolving graph structure;
- real-time suspicious trader detection.

### Vai trò trong thesis

Dùng để hỗ trợ hai luận điểm:

1. fraud detection nên được xem như bài toán dynamic graph thay vì static graph;
2. online/early detection cần evaluation khác với offline binary classification.

### Thesis kế thừa

- streaming perspective;
- temporal evolution;
- online anomaly viewpoint.

### Thesis khác biệt

FiFrauD không mô hình hóa riêng mule-account lifecycle và role transitions.

### Mức ưu tiên

- P1 research anchor;
- implementation end-to-end là **optional**;
- có thể dùng unsupervised score như một baseline mở rộng nếu tiến độ cho phép.

---

# 7. Các công trình gần đây dùng để xác lập novelty và mở rộng

## 7.1. Spade+ — real-time dynamic graph system

**Spade+: A Generic Real-Time Fraud Detection Framework on Dynamic Graphs.**  
*IEEE Transactions on Knowledge and Data Engineering*, Vol. 36, No. 11, pp. 7058–7073, 2024.  
DOI: <https://doi.org/10.1109/TKDE.2024.3394155>

Vai trò:

- định hướng streaming/dynamic graph implementation;
- tham khảo latency và dynamic edge management;
- không cần reproduce toàn bộ.

---

## 7.2. SEFraud — explainable graph fraud detection

**SEFraud: Graph-based Self-Explainable Fraud Detection via Interpretative Mask Learning.**  
*ACM SIGKDD 2024*, pp. 5329–5338.  
DOI: <https://doi.org/10.1145/3637528.3671534>

Vai trò:

- tham khảo explainability;
- optional extension nếu phần core đã hoàn tất.

---

## 7.3. Grad — fraud camouflage

**Grad: Guided Relation Diffusion Generation for Graph Augmentation in Graph Fraud Detection.**  
*Proceedings of the ACM Web Conference 2025*, pp. 5308–5319.  
DOI: <https://doi.org/10.1145/3696410.3714520>

Code: <https://github.com/Muyiiiii/WWW25-Grad>

Vai trò:

- chỉ ra vấn đề **adaptive camouflage**;
- có thể dùng để thiết kế robustness test;
- không phải baseline bắt buộc trong 4 tháng.

---

## 7.4. TF-GNN — recent heterogeneous temporal fraud anchor

**TF-GNN: A Temporal Feedback Graph Neural Network with Discriminative Regularization for Heterogeneous Fraud Detection.**  
*Neurocomputing*, Vol. 687, Article 133694, 2026.  
DOI: <https://doi.org/10.1016/j.neucom.2026.133694>

Vai trò:

- cập nhật hướng dynamic heterogeneous fraud detection mới;
- tham khảo historical feedback;
- dùng trong related work và discussion.

---

# 8. Baseline thực nghiệm cần chạy trên cùng dataset

Research anchors ở trên dùng để xác lập novelty. Comparison chính phải được thực hiện trên D1 với cùng target và cùng temporal protocol.

## B0. Rule/Heuristic Controls

Tối thiểu một số mule rules:

- rapid pass-through;
- fan-in threshold;
- fan-out threshold;
- short dwell-time;
- high outflow/inflow ratio;
- high new-counterparty ratio.

Mục tiêu:

> Xác định mức hiệu quả của các typology rules trước khi dùng ML/GNN.

---

## B1. Tabular ML

Chọn:

- **LightGBM** hoặc
- **XGBoost**.

Input:

\[
x_v(t)
\]

gồm các point-in-time behavioral features.

Không dùng graph embedding.

Mục tiêu:

> Kiểm tra temporal/graph model có thực sự vượt strong tabular fraud baseline hay không.

---

## B2. Static GNN

Chọn một:

- GraphSAGE;
- GAT.

Graph được xây từ training-period hoặc từ snapshot tại thời điểm đánh giá.

Không dùng explicit event-time message passing.

Mục tiêu:

\[
Graph\ value
\]

so với tabular features.

---

## B3. Temporal GNN

Chọn một backbone chính:

- **TGN** — ưu tiên; hoặc
- TGAT.

Input:

\[
e_k=(u,v,t_k,x_k).
\]

Mục tiêu:

\[
Temporal\ graph\ value
\]

so với static GNN.

---

## B4. STA-GT-inspired baseline — BẮT BUỘC

Dùng:

- graph neighborhood;
- temporal encoding;
- global attention/transformer.

Không dùng:

- multi-scale mule role encoder;
- role-transition module.

B4 là recent top-tier baseline quan trọng nhất.

---

## B5. MLaD²-style baseline — P1

Nếu implementation full quá nặng, dùng phiên bản thích nghi:

\[
GraphEncoder
+
ContrastivePretraining
+
SupervisedFineTune.
\]

Không có role-transition branch.

B5 ưu tiên cho label-scarcity study.

---

## B6. FiFrauD-style unsupervised baseline — OPTIONAL

Dùng như:

- unsupervised anomaly rank;
- comparison trong zero/few-label scenario.

Không để B6 làm chậm phần core.

---

## B7. Proposed EarlyMule-TG

\[
TemporalGraphEncoder
+
MultiScaleRoleEncoder
+
RoleTransitionFusion
\rightarrow
MuleRiskScore.
\]

---

# 9. Ba bộ dữ liệu

## D1. AMLSim-Mule — P0, BẮT BUỘC

Nguồn nền:

**IBM AMLSim**  
<https://github.com/IBM/AMLSim>

AMLSim là multi-agent simulator sinh:

- synthetic banking accounts;
- transactions;
- known laundering patterns;
- alert accounts;
- alert transactions.

### Vai trò

D1 là dataset chính để:

- định nghĩa rõ mule role;
- định nghĩa mule onset;
- định nghĩa completion/cash-out time;
- đánh giá early detection;
- ablation;
- label-scarcity;
- typology-specific recall.

### Phần thesis phải bổ sung

Xây cấu hình:

\[
\textbf{AMLSim-Mule}
\]

với tối thiểu bốn typology.

### Typology M1 — Fan-in → rapid fan-out

```text
Source_1 ─┐
Source_2 ─┼──> Mule ─┬──> Dest_1
Source_3 ─┘          ├──> Dest_2
                     └──> Dest_3
```

### Typology M2 — Relay chain

```text
Source -> Mule_1 -> Mule_2 -> Mule_3 -> Cashout
```

### Typology M3 — Collector → single exit

```text
Many Sources -> Collector Mule -> Exit Account
```

### Typology M4 — Collector → relay mules → dispersal

```text
Sources
   │
   ▼
Collector
   │
   ├──> Relay_1 ──> Dest_1
   ├──> Relay_2 ──> Dest_2
   └──> Relay_3 ──> Dest_3
```

### Optional M5

Cycle/ring laundering.

### Ground truth cần lưu

```text
mule_id
typology
role
mule_start_time
first_illicit_inflow_time
first_illicit_outflow_time
completion_time
cashout_time
upstream_accounts
downstream_accounts
```

### Đây là điểm quan trọng

Không chỉ lưu:

```text
is_mule = 1
```

mà phải có **temporal ground truth** để đo early detection.

---

## D2. IBM Transactions for Anti Money Laundering — P1

Nguồn:

**IBM AML-Data**  
<https://github.com/IBM/AML-Data>

Paper nền:

**Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.**  
*NeurIPS 2023.*

Nguồn IBM Research:  
<https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models>

### Nội dung

Dataset mô phỏng:

- banks;
- individuals;
- companies;
- transfers;
- payment types;
- laundering transactions.

Dataset có timestamp và laundering labels.

### Vai trò

D2 dùng để kiểm tra:

> Phương pháp có còn hiệu quả khi chuyển từ mule-specific simulation sang một AML transaction network có quy mô lớn và pattern đa dạng hơn hay không?

### Lưu ý

D2 không cung cấp hoàn toàn cùng semantic “money mule” như D1.

Do đó:

- primary claim vẫn dựa trên D1;
- D2 dùng cho **AML-account / laundering-participant generalization**;
- phải ghi rõ cách chuyển transaction label thành account-level target.

### Scope đề xuất

Bắt đầu bằng subset:

- LI-Small hoặc HI-Small;
- sau đó mở rộng nếu compute cho phép.

---

## D3. Elliptic++ Actors — P2, OPTIONAL EXTERNAL VALIDATION

Nguồn:

<https://github.com/git-disl/EllipticPlusPlus>

Actors dataset có:

- 822,942 wallet addresses;
- 1,268,260 temporal interactions;
- 2,868,964 address-address edges;
- 49 time steps;
- licit / illicit / unknown labels.

### Vai trò

D3 dùng để kiểm tra:

\[
Temporal\ account\ representation
\]

trên financial transaction network khác miền.

### Lưu ý bắt buộc

**Không gọi D3 là digital banking dataset.**

Đây là:

> Bitcoin blockchain illicit-address benchmark.

D3 chỉ hỗ trợ external validity.

---

# 10. Problem Formulation

Cho dòng giao dịch:

\[
\mathcal{E}
=
\{e_1,e_2,\dots,e_N\}
\]

với:

\[
e_i=
(u_i,v_i,t_i,a_i,x_i),
\]

trong đó:

- \(u_i\): source account;
- \(v_i\): destination account;
- \(t_i\): timestamp;
- \(a_i\): amount;
- \(x_i\): transaction metadata.

Tại thời điểm \(t\), chỉ được sử dụng:

\[
\mathcal{H}_{\le t}
=
\{e_i:t_i\le t\}.
\]

Temporal transaction graph:

\[
G(t)
=
(V(t),E_{\le t}).
\]

Với account \(v\), mô hình dự đoán:

\[
\hat p_v(t)
=
P(y_v^{mule}=1
\mid
G(t),\mathcal{H}_{\le t}).
\]

Điều kiện bắt buộc:

\[
\boxed{
No\ feature\ may\ use\ information\ after\ t.
}
\]

---

# 11. Định nghĩa Early Detection

Với mỗi mule account \(v\):

- \(\tau_v^{start}\): thời điểm bắt đầu mule activity;
- \(\tau_v^{complete}\): thời điểm chuỗi laundering/mule activity hoàn thành;
- \(\tau_v^{detect}\): thời điểm đầu tiên model vượt alert threshold.

## Detection delay

\[
TTD_v
=
\tau_v^{detect}
-
\tau_v^{start}.
\]

## Detection lead time

\[
LeadTime_v
=
\tau_v^{complete}
-
\tau_v^{detect}.
\]

Mục tiêu:

\[
TTD_v \downarrow
\]

và:

\[
LeadTime_v \uparrow.
\]

Một mule được xem là **early-detected** nếu:

\[
\tau_v^{detect}
<
\tau_v^{complete}.
\]

Có thể thêm ngưỡng:

\[
\tau_v^{detect}
\le
\tau_v^{complete}-\Delta.
\]

Ví dụ:

- \(\Delta=30\) phút;
- \(\Delta=1\) giờ;
- hoặc theo quantile của typology duration.

---

# 12. Point-in-Time Feature Construction

Tại thời điểm \(t\), mọi feature phải được tính từ trailing window:

\[
[t-w,t].
\]

Không được dùng full-history aggregate chứa future transactions.

## 12.1. Các cửa sổ thời gian

MVP:

\[
W=\{1h,24h,7d\}.
\]

Nếu D1 có timeline ngắn hơn, điều chỉnh thành:

\[
W=\{30m,6h,24h\}.
\]

## 12.2. Basic features

Theo mỗi \(w\):

- number of inbound transactions;
- number of outbound transactions;
- inbound amount;
- outbound amount;
- mean/median/max amount;
- unique inbound counterparties;
- unique outbound counterparties;
- transaction frequency;
- inter-arrival statistics.

## 12.3. Mule-behavior features

### Pass-through ratio

\[
PTR_v^{(w)}
=
\frac{OutAmount_v^{(w)}}
{InAmount_v^{(w)}+\epsilon}.
\]

### Fan-in

\[
FI_v^{(w)}
=
|\mathcal{N}_{in}^{(w)}(v)|.
\]

### Fan-out

\[
FO_v^{(w)}
=
|\mathcal{N}_{out}^{(w)}(v)|.
\]

### Counterparty churn

\[
CCR_v^{(w)}
=
\frac{\#NewCounterparties}
{\#Counterparties+\epsilon}.
\]

### Burstiness

Dùng inter-arrival time:

\[
B=
\frac{\sigma_\Delta-\mu_\Delta}
{\sigma_\Delta+\mu_\Delta}.
\]

### Flow imbalance

\[
I_v^{(w)}
=
\frac{OutAmount-InAmount}
{OutAmount+InAmount+\epsilon}.
\]

### Rapid-forward ratio

Tỷ lệ tiền vào được chuyển tiếp trong một khoảng \(\delta\).

### Dwell-time statistics

- median dwell time;
- min dwell time;
- fraction forwarded within 30m/1h/6h.

### Two-hop flow intensity

Đo mức:

\[
Upstream
\rightarrow
v
\rightarrow
Downstream.
\]

### Counterparty entropy

\[
H_{cp}
=
-\sum_i p_i\log p_i.
\]

---

# 13. Multi-Scale Mule Role Representation

Đây là contribution trung tâm thứ nhất.

Với mỗi cửa sổ:

\[
w\in W,
\]

xây behavioral vector:

\[
b_v^{(w)}(t).
\]

Role encoder:

\[
r_v^{(w)}(t)
=
MLP_w(b_v^{(w)}(t)).
\]

Multi-scale fusion:

\[
r_v(t)
=
Attention(
r_v^{(w_1)},
r_v^{(w_2)},
...,
r_v^{(w_m)}
).
\]

Không hard-code node thành một class duy nhất.

Role embedding được kỳ vọng phản ánh continuum:

```text
Normal
  ↓
Receiver
  ↓
Collector
  ↓
Relay
  ↓
Distributor
```

---

# 14. Role-Transition Representation

Đây là contribution trung tâm thứ hai.

Không chỉ học trạng thái:

\[
r_v(t),
\]

mà học sự thay đổi:

\[
\Delta r_v(t)
=
r_v(t)
-
r_v(t-\delta).
\]

Hoặc với nhiều scale:

\[
\Delta r_v^{(i,j)}
=
r_v^{(w_i)}
-
r_v^{(w_j)}.
\]

Role-transition vector:

\[
q_v(t)
=
TransitionEncoder(
r_v(t),
\Delta r_v(t)
).
\]

Mục tiêu:

> Bắt sự chuyển đổi đột ngột từ normal receiver sang pass-through relay/distributor.

Ví dụ:

```text
t0: low activity / normal
t1: sudden fan-in
t2: very short dwell
t3: fan-out
```

Pattern:

\[
Normal
\rightarrow
Collector
\rightarrow
Relay/Distributor.
\]

---

# 15. Temporal Graph Encoder

## 15.1. Backbone chính

Khuyến nghị:

**TGN**.

Lý do:

- event-based;
- phù hợp transaction streams;
- có node memory;
- dễ triển khai point-in-time;
- phù hợp account-level dynamic state.

Output:

\[
h_v^{TG}(t).
\]

## 15.2. Alternative

Nếu TGN quá khó:

- TGAT.

Không đổi research question.

## 15.3. Edge representation

\[
x_e=
[
\log(1+amount),
transaction\_type,
direction,
\Delta t,
time\_of\_day,
day\_of\_week
].
\]

Chỉ dùng feature có sẵn trước hoặc tại transaction time.

---

# 16. Proposed Architecture — EarlyMule-TG

Kiến trúc tổng thể:

```text
     Transaction Stream                Transaction History
              │                                 │
              ▼                                 ▼
 ┌────────────────────────┐      ┌────────────────────────────┐
 │ Temporal Graph Encoder │      │   Multi-scale Behavioral   │
 │       TGN / TGAT       │      │          Features          │
 └────────────┬───────────┘      └──────────────┬─────────────┘
              │                                 │
              │                                 ▼
              │                  ┌────────────────────────────┐
              │                  │  Multi-Scale Role Encoder  │
              │                  └──────────────┬─────────────┘
              │                                 │
              │                                 │ r_v(t)
              │ h_v^TG(t)                       ▼
              │                  ┌────────────────────────────┐
              │                  │  Role Transition Encoder   │
              │                  └──────────────┬─────────────┘
              │                                 │
              │                                 │
              │                                 │ q_v(t)
              │ (Temporal Graph)                │ (Role Dynamics)
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Gated / Attention Fusion │
                 └─────────────┬────────────┘
                               │
                               ▼
                        Mule Risk Score
```

## 16.1. Fusion

\[
z_v(t)
=
Fuse(
h_v^{TG}(t),
r_v(t),
q_v(t)
).
\]

MVP:

\[
g_v
=
\sigma(
W_g[
h_v^{TG};
r_v;
q_v
]
).
\]

\[
z_v
=
g_v\odot h_v^{TG}
+
(1-g_v)\odot [r_v;q_v].
\]

Classifier:

\[
\hat y_v(t)
=
\sigma(
MLP(z_v(t))
).
\]

---

# 17. Không nên làm architecture inflation

Không thêm module chỉ để tăng độ phức tạp.

Thứ tự ưu tiên contribution:

1. **point-in-time temporal graph protocol**;
2. **multi-scale mule behavioral representation**;
3. **role-transition representation**;
4. early-detection evaluation;
5. label-scarcity study.

Nếu ba thành phần đầu không tạo gain rõ, không thêm:

- diffusion;
- LLM;
- GAN;
- federated learning;
- RL;
- hypergraph;
- extra transformer blocks.

Ưu tiên:

\[
Sound\ Evaluation
>
Complex\ Architecture.
\]

---

# 18. Xử lý mất cân bằng lớp

Mule accounts thường chiếm tỷ lệ rất nhỏ.

## 18.1. Loss cơ bản

Weighted BCE:

\[
L_{cls}
=
-w_+y\log \hat y
-
w_-(1-y)\log(1-\hat y).
\]

## 18.2. Alternative

Focal loss:

\[
L_{focal}
=
-\alpha(1-\hat p_t)^\gamma\log \hat p_t.
\]

## 18.3. Không SMOTE trên graph trước split

Không thực hiện:

```text
SMOTE -> split
```

vì có nguy cơ leakage.

Nếu cần oversampling:

- chỉ trên training set;
- hoặc sample positive nodes/events khi tạo minibatch.

## 18.4. Không tối ưu Accuracy

Accuracy chỉ là secondary metric.

---

# 19. Label-Scarcity Protocol

Từ training labels:

\[
L
\]

lấy các mức:

\[
\{1\%,5\%,10\%,25\%,100\%\}.
\]

Giữ nguyên:

- validation set;
- test set;
- graph chronology.

So sánh tối thiểu:

- LightGBM;
- TGN;
- STA-GT-inspired;
- EarlyMule-TG.

Nếu triển khai MLaD²-style pretraining:

- thêm vào bảng này.

Mỗi mức label:

- 3 seeds nếu compute đủ;
- báo mean ± std.

---

# 20. Objective Functions

## 20.1. Classification loss

\[
L_{cls}.
\]

## 20.2. Optional role-transition consistency loss

Khuyến khích các account có mule transition tương tự có representation gần nhau:

\[
L_{role}.
\]

Chỉ thêm nếu:

- có ground-truth typology;
- implementation ổn định.

## 20.3. Optional contrastive pretraining

\[
L_{ssl}.
\]

Dùng cho MLaD²-style extension.

## 20.4. Tổng loss

MVP:

\[
L
=
L_{cls}.
\]

Full optional:

\[
L
=
L_{cls}
+
\lambda_rL_{role}
+
\lambda_sL_{ssl}.
\]

Không bắt buộc dùng nhiều loss nếu không chứng minh được lợi ích.

---

# 21. Evaluation Protocol

## 21.1. Không random split

Bắt buộc:

\[
Train
<
Validation
<
Test.
\]

Ví dụ:

```text
Train:      0%  ─────────  60%   (timeline)
Validation: 60% ─────────  75%
Test:       75% ───────── 100%
```

Hoặc split theo ngày/tuần tùy dataset.

## 21.2. Point-in-time graph

Khi dự đoán tại \(t\):

\[
E_{model}
=
\{e:t_e\le t\}.
\]

Không được:

- build graph từ toàn bộ dataset rồi mask label;
- tính degree bằng future edges;
- tính global account statistics sau test time;
- normalize bằng test distribution.

## 21.3. Negative sampling

Negative account tại \(t\) phải là account chưa được xác định positive tại thời điểm đang xét.

Cần tránh:

> account về sau trở thành mule nhưng được dùng như benign training sample sau thời điểm đã có dấu hiệu illicit.

Phải định nghĩa policy rõ ràng.

## 21.4. Chronological folds

Tối thiểu:

- 3 temporal folds hoặc
- 1 fixed temporal holdout + 3 seeds nếu dataset khó chia.

Khuyến nghị:

```text
Fold 1: Train T0–T1 | Val T1–T2 | Test T2–T3
Fold 2: Train T0–T2 | Val T2–T3 | Test T3–T4
Fold 3: Train T0–T3 | Val T3–T4 | Test T4–T5
```

## 21.5. Typology holdout — mở rộng mạnh

Có thể thử:

```text
Train: M1, M2, M3
Test:  M4
```

để đánh giá khả năng phát hiện unseen laundering pattern.

Chỉ làm sau main experiment.

---

# 22. Leakage Audit bắt buộc

Phải tạo:

`data_leakage_audit.md`

Kiểm tra ít nhất:

1. transaction timestamp;
2. future degree;
3. future counterparty count;
4. full-history amount statistics;
5. normalization leakage;
6. positive-label onset leakage;
7. graph construction leakage;
8. duplicate transaction leakage;
9. account overlap policy;
10. typology metadata leakage;
11. simulator-only fields;
12. future balance leakage;
13. feature computation cache leakage.

Đặc biệt:

> Không dùng các trường chỉ simulator mới biết như `is_sar`, `alert_id`, `pattern_id`, `mule_role` làm input model.

Các trường đó chỉ được dùng làm **ground truth/evaluation metadata**.

---

# 23. Metrics

## 23.1. Primary metrics

### PR-AUC / Average Precision

Đây là metric chính do class imbalance.

### Recall@K

Với alert budget \(K\):

\[
Recall@K
=
\frac{
\#Mules\ in\ top\ K
}{
\#All\ Mules
}.
\]

Thử:

- K = 50;
- K = 100;
- hoặc top 0.1%, 0.5%, 1%.

### Precision@K

Đo operational usefulness.

## 23.2. Secondary metrics

- ROC-AUC;
- F1;
- F1-Macro;
- MCC;
- Balanced Accuracy.

## 23.3. Early-detection metrics

- Time-to-Detect;
- Detection Lead Time;
- Early Recall;
- Early Precision;
- fraction detected before completion.

## 23.4. Efficiency

Nếu thời gian cho phép:

- inference latency;
- events/second;
- accounts/second;
- GPU memory;
- model parameters.

---

# 24. Early-Detection Evaluation

Với mỗi positive account:

```text
Start ──────────────────────────────── Completion
  │                                       │
  └──── model can accumulate evidence ────┘
```

Mỗi thời điểm sau start:

\[
t_1,t_2,\dots,t_m
\]

model phát:

\[
p_v(t_i).
\]

Threshold:

\[
\theta
\]

được chọn **trên validation set**.

Detection time:

\[
\tau_v^{detect}
=
\min\{t:p_v(t)\ge\theta\}.
\]

Không chọn threshold trên test set.

## Báo cáo theo typology

| Typology | Recall | Median TTD | Median Lead Time | Precision@K |
|---|---:|---:|---:|---:|
| M1 Fan-in/out | | | | |
| M2 Relay | | | | |
| M3 Collector | | | | |
| M4 Dispersal | | | | |

---

# 25. Ablation Study bắt buộc

## A0 — Tabular

\[
BehaviorFeatures
\rightarrow
LightGBM.
\]

## A1 — Static Graph

\[
StaticGNN.
\]

## A2 — Temporal Graph

\[
TGN.
\]

## A3 — Temporal Graph + Multi-Scale Behavior

\[
TGN
+
MultiScaleBehavior.
\]

## A4 — Temporal Graph + Role State

\[
TGN
+
RoleEncoder.
\]

## A5 — Temporal Graph + Role Transition

\[
TGN
+
RoleEncoder
+
TransitionEncoder.
\]

## A6 — Full EarlyMule-TG

\[
TemporalGraph
+
MultiScaleRole
+
RoleTransition
+
Fusion.
\]

Bảng chuẩn:

| Model | Tabular | Graph | Temporal | Multi-scale | Role State | Role Transition |
|---|---:|---:|---:|---:|---:|---:|
| A0 LightGBM | ✓ | | | ✓ | | |
| A1 GraphSAGE/GAT | ✓ | ✓ | | | | |
| A2 TGN | ✓ | ✓ | ✓ | | | |
| A3 TGN + Multi-scale | ✓ | ✓ | ✓ | ✓ | | |
| A4 TGN + Role | ✓ | ✓ | ✓ | ✓ | ✓ | |
| A5 TGN + Transition | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **A6 EarlyMule-TG** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

A5/A6 có thể khác nhau ở fusion hoặc full loss.

---

# 26. Baseline Comparison Table cần có trong khóa luận

| Model | Venue/Origin | Graph | Temporal | Account-level | Few-label | Early-detection | Role transition |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | classical | | ✓* | ✓ | | | |
| GraphSAGE/GAT | classical GNN | ✓ | | ✓ | | | |
| TGN/TGAT | temporal GNN | ✓ | ✓ | ✓ | | | |
| STA-GT-inspired | IEEE TII 2024 | ✓ | ✓ | adapt | | | |
| MLaD²-style | IEEE TIFS 2024 | ✓ | limited | ✓ | ✓ | | |
| FiFrauD-style | ACM TKDD 2024 | ✓ | ✓ | ✓ | label-free | online | |
| **EarlyMule-TG** | thesis | ✓ | ✓ | ✓ | evaluate | ✓ | ✓ |

\* temporal aggregates nhưng không học graph dynamics.

---

# 27. Success Criteria / KPI

## 27.1. Mức tối thiểu để hoàn thành khóa luận tốt

1. D1 AMLSim-Mule chạy ổn định và tái lập được.
2. Có tối thiểu 4 mule typologies.
3. Có temporal ground truth cho mule onset/completion.
4. Có leakage-safe point-in-time pipeline.
5. Có:
   - LightGBM/XGBoost;
   - static GNN;
   - TGN/TGAT;
   - STA-GT-inspired baseline.
6. EarlyMule-TG train/evaluate end-to-end.
7. Có PR-AUC và Recall@K.
8. Có Time-to-Detect / Lead Time.
9. Có ít nhất 4 ablation configurations.
10. Có code/config/README tái lập được.

## 27.2. Mức tốt

Ngoài yêu cầu trên:

- Proposed cải thiện trung bình khoảng ≥3% PR-AUC hoặc Recall@K so với strong temporal baseline;
- hoặc giảm median TTD ≥10%;
- gain xuất hiện trên ≥2 mule typologies;
- label-scarcity experiment đầy đủ;
- có bootstrap confidence interval.

**Lưu ý:** các ngưỡng 3% và 10% là KPI quản trị tiến độ, không phải điều kiện để kết luận nghiên cứu hợp lệ.

Negative result vẫn chấp nhận được nếu:

- protocol đúng;
- không leakage;
- analysis rõ;
- kết quả tái lập được.

## 27.3. Mức rất tốt / có tiềm năng bài báo

- external validation trên D2;
- EarlyMule-TG tăng Recall@K và giảm TTD rõ;
- role-transition ablation có statistical significance;
- label-scarcity performance tốt;
- unseen-typology experiment;
- có case study trực quan;
- code/data-generation pipeline reproducible.

---

# 28. Kế hoạch 4 tháng / 16 tuần

# THÁNG 1 — Literature, data, ground truth, leakage-safe baselines

## Tuần 1 — Khóa bài toán và đọc anchor papers

Đọc sâu:

- STA-GT;
- MLaD²;
- FiFrauD;
- lướt Spade+, Grad, TF-GNN.

Deliverables:

- `literature_notes.md`;
- bảng comparison;
- finalized RQ/Hypotheses;
- repository skeleton;
- environment chạy được.

**Gate G1:** Không đổi core thesis sau tuần 2 trừ khi D1 không khả thi.

---

## Tuần 2 — AMLSim ingestion + mule typology design

Thực hiện:

- chạy AMLSim;
- kiểm tra accounts;
- transactions;
- alert accounts;
- alert transactions;
- hiểu simulator pattern mechanism.

Thiết kế:

- M1;
- M2;
- M3;
- M4.

Deliverables:

```text
mule_typologies.md
simulation_config/
raw_transactions.csv
mule_ground_truth.csv
```

---

## Tuần 3 — Temporal ground truth + leakage audit

Xây:

```text
mule_id
role
start_time
completion_time
typology
```

Thực hiện:

- chronological split;
- point-in-time feature functions;
- leakage audit.

Deliverable:

`data_leakage_audit.md`

**Gate G2:** Không bắt đầu model phức tạp khi temporal ground truth chưa kiểm chứng.

---

## Tuần 4 — B0 + B1

Chạy:

- typology rules;
- LightGBM/XGBoost.

Feature groups:

- amount;
- velocity;
- fan-in/fan-out;
- counterparties;
- pass-through;
- dwell-time;
- burstiness.

Deliverable:

`tabular_baseline.md`

---

# THÁNG 2 — Static/temporal graph baselines

## Tuần 5 — Static GraphSAGE/GAT

Build account graph.

Chạy:

- GraphSAGE hoặc GAT.

So sánh:

\[
B1\quad vs\quad B2.
\]

Deliverables:

- graph builder;
- static baseline;
- degree/topology EDA.

---

## Tuần 6 — TGN/TGAT

Implement:

\[
(u,v,t,x_e)
\]

event stream.

Chạy first temporal model.

Sanity checks:

- memory updates only after event;
- no future edges;
- correct negative sampling;
- correct account state time.

**Gate G3:** TGN phải chạy end-to-end trước khi phát triển proposed model.

---

## Tuần 7 — STA-GT-inspired baseline

Implement:

- temporal encoding;
- graph neighborhood;
- attention/global interaction.

Không thêm role-transition features.

Deliverable:

`sta_gt_adaptation.md`

Phải ghi:

- phần nào giữ từ paper;
- phần nào điều chỉnh;
- lý do điều chỉnh.

---

## Tuần 8 — Multi-scale behavioral feature module

Xây:

\[
W=\{1h,24h,7d\}
\]

hoặc cửa sổ phù hợp D1.

Implement:

```python
build_point_in_time_features(account_id, timestamp)
```

Deliverables:

- feature module;
- feature tests;
- 100 manually inspected account-time samples;
- typology-feature visualization.

---

# THÁNG 3 — Role-transition model

## Tuần 9 — Role Encoder

Xây:

\[
b_v^{(w)}
\rightarrow
r_v^{(w)}
\rightarrow
r_v.
\]

Visualize:

- normal;
- collector;
- relay;
- distributor.

Không yêu cầu cluster tách hoàn hảo.

Yêu cầu:

- representation có variation hợp lý;
- không encode simulator label.

---

## Tuần 10 — Role Transition Encoder

Xây:

\[
q_v(t)
=
TransitionEncoder(
r_v(t),
r_v(t-\delta)
).
\]

Case study:

```text
normal -> receiver -> collector -> relay
```

Deliverables:

- transition module;
- transition curves;
- typology-level examples.

---

## Tuần 11 — EarlyMule-TG integration

Kết hợp:

\[
h_v^{TG}
+
r_v
+
q_v.
\]

Implement:

- gated fusion hoặc attention fusion;
- weighted BCE/focal loss.

Sanity tests:

- no NaN;
- stable gradients;
- role branch không collapse;
- temporal graph branch hoạt động độc lập.

---

## Tuần 12 — Preliminary ablation

Chạy:

- A0;
- A1;
- A2;
- A3;
- A5/A6.

**Gate G4:** Cuối tuần 12 phải có bảng preliminary results.

Nếu Proposed không hơn temporal baseline:

1. không thêm architecture;
2. kiểm tra leakage;
3. kiểm tra threshold;
4. kiểm tra role features;
5. kiểm tra typology balance;
6. phân tích lỗi.

---

# THÁNG 4 — Frozen evaluation, early detection, writing

## Tuần 13 — Freeze main experiment

Freeze:

- dataset version;
- simulator seed set;
- train/val/test split;
- features;
- metrics;
- threshold selection;
- configs.

Chạy:

- main baselines;
- Proposed;
- 3 seeds nếu compute đủ.

Không thay test set sau khi nhìn kết quả.

---

## Tuần 14 — Early detection + label scarcity + ablation

Hoàn thành:

- Recall@K;
- Precision@K;
- TTD;
- Lead Time;
- typology breakdown;
- label 1/5/10/25/100%;
- ablation;
- bootstrap CI nếu đủ.

Nếu xong sớm:

- bắt đầu D2.

---

## Tuần 15 — External validation + viết khóa luận

Ưu tiên:

1. D2 IBM AML;
2. nếu D2 không xong thì không chạy D3.

Viết:

1. Introduction
2. Background and Related Work
3. Problem Formulation
4. Proposed Method
5. Experimental Setup
6. Results and Discussion
7. Conclusion and Future Work

---

## Tuần 16 — Finalization

Hoàn thành:

- thesis PDF;
- LaTeX source;
- source code;
- README;
- environment file;
- configs;
- simulator configs;
- experiment logs;
- figures;
- slides;
- demo notebook;
- reproducibility run.

---

# 29. Cấu trúc repository đề xuất

```text
earlymule-tg/
├── README.md
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── simulation/
│   │   ├── m1_fanin_fanout.yaml
│   │   ├── m2_relay.yaml
│   │   ├── m3_collector.yaml
│   │   └── m4_dispersal.yaml
│   │
│   ├── lightgbm.yaml
│   ├── graphsage.yaml
│   ├── tgn.yaml
│   ├── sta_gt.yaml
│   └── earlymule_tg.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── amlsim.py
│   │   ├── ibm_aml.py
│   │   ├── ground_truth.py
│   │   ├── splits.py
│   │   └── leakage.py
│   │
│   ├── features/
│   │   ├── point_in_time.py
│   │   ├── behavioral.py
│   │   ├── flow.py
│   │   └── motifs.py
│   │
│   ├── graph/
│   │   ├── static_graph.py
│   │   ├── temporal_events.py
│   │   └── sampler.py
│   │
│   ├── models/
│   │   ├── tabular.py
│   │   ├── graphsage.py
│   │   ├── tgn.py
│   │   ├── sta_gt.py
│   │   ├── role_encoder.py
│   │   ├── transition_encoder.py
│   │   └── earlymule_tg.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── early_detection.py
│   │   ├── label_scarcity.py
│   │   └── bootstrap.py
│   │
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
│   ├── baselines/
│   ├── ablation/
│   ├── early_detection/
│   ├── label_scarcity/
│   └── external_validation/
│
├── notebooks/
│
├── reports/
│   ├── literature_notes.md
│   ├── mule_typologies.md
│   ├── data_leakage_audit.md
│   ├── sta_gt_adaptation.md
│   └── experiment_log.md
│
└── thesis/
    ├── main.tex
    └── figures/
```

---

# 30. Reproducibility requirements

Mỗi experiment phải lưu:

```text
experiment_id
git_commit
dataset_name
dataset_version
simulation_seed
dataset_seed
train_period
validation_period
test_period
random_seed
model_config
feature_config
window_config
label_fraction
threshold
metrics
runtime
```

Command-line tối thiểu:

```bash
python src/train.py --config configs/tgn.yaml
python src/train.py --config configs/earlymule_tg.yaml
python src/evaluate.py --checkpoint <path>
```

Không chấp nhận kết quả cuối chỉ tồn tại trong notebook.

---

# 31. Experiment Table cần duy trì

`experiments.csv`

| Exp ID | Dataset | Fold | Model | Labels | Seed | PR-AUC | ROC-AUC | Recall@K | Precision@K | TTD | Lead Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

Mục tiêu:

- truy vết tuning;
- tránh cherry-picking;
- tái lập được result tables;
- phân biệt development và frozen test.

---

# 32. Figures bắt buộc trong khóa luận

1. Overall EarlyMule-TG architecture.
2. Mule typology diagrams M1–M4.
3. Temporal train/validation/test split.
4. Point-in-time feature construction.
5. Multi-scale role representation.
6. Example role-transition trajectory.
7. PR curve.
8. Recall@K curve.
9. Early-detection timeline.
10. Case-study transaction subgraph.

---

# 33. Tables bắt buộc

1. Related-work comparison.
2. Research-anchor comparison.
3. Dataset statistics.
4. Mule typology statistics.
5. Main baseline comparison.
6. Ablation study.
7. Label-scarcity results.
8. Per-typology results.
9. Early-detection results.
10. Runtime/efficiency summary nếu có.

---

# 34. Error Analysis bắt buộc

Phân tích tối thiểu:

## False Positive

Các account:

- high-volume merchant;
- salary/payroll account;
- payment aggregator;
- family-transfer hub;
- legitimate bursty account.

Câu hỏi:

> Tại sao chúng giống mule?

## False Negative

Các mule:

- slow laundering;
- low-volume mule;
- long dwell-time;
- one-to-one relay;
- camouflage transactions.

Câu hỏi:

> Role-transition module bỏ sót pattern nào?

---

# 35. Case Study

Chọn tối thiểu:

- 2 true positive;
- 2 false positive;
- 2 false negative.

Với mỗi case:

```text
Account timeline
Transaction subgraph
Feature trajectory
Risk-score trajectory
Detected time
Completion time
Role-transition interpretation
```

Không chỉ đưa confusion matrix.

---

# 36. Statistical Analysis

Nếu compute/sample đủ:

- 3 seeds;
- bootstrap 95% CI;
- paired bootstrap cho PR-AUC/Recall@K;
- significance test giữa A2 và A6.

Không cần ép p-value nếu sample không phù hợp.

Quan trọng hơn:

- effect size;
- consistency across folds;
- consistency across typologies.

---

# 37. Rủi ro và fallback plan

## R1. AMLSim khó cấu hình mule role chính xác

### Xử lý

Giữ ít nhất:

- M1 fan-in/fan-out;
- M2 relay chain;
- M3 collector.

Không cố thêm quá nhiều typology.

Mục tiêu là:

> ground truth rõ hơn số lượng pattern.

---

## R2. TGN quá phức tạp hoặc lỗi memory

### Xử lý

Fallback:

- TGAT;
- hoặc temporal snapshot GraphSAGE.

Không quay về random static split.

---

## R3. STA-GT không reproduce nguyên gốc

### Xử lý

Dùng STA-GT-inspired adaptation.

Phải ghi rõ:

- paper component;
- adaptation;
- implementation difference.

Không claim exact reproduction.

---

## R4. Role encoder không tạo gain

### Xử lý

Kiểm tra:

- feature scaling;
- window size;
- temporal alignment;
- mule typology quality;
- role transition definition.

Nếu vẫn không gain:

- giữ negative result;
- phân tích typology nào temporal GNN đã đủ;
- không thêm module phức tạp để “cứu” kết quả.

---

## R5. Proposed tăng PR-AUC nhưng không giảm TTD

### Xử lý

Đây là kết quả nghiên cứu có ý nghĩa.

Kết luận:

> model tốt hơn về ranking nhưng chưa tốt hơn về early intervention.

Phân tích risk trajectory và threshold.

---

## R6. D2 quá lớn

### Xử lý

Dùng:

- LI-Small hoặc HI-Small;
- subset thời gian;
- streaming preprocessing;
- không load toàn dataset vào RAM.

D2 không được làm chậm D1.

---

## R7. D3 không phù hợp semantic mule account

### Xử lý

Không sử dụng nếu dễ gây lệch claim.

D3 chỉ là optional financial temporal-graph validation.

---

## R8. Class imbalance quá cực đoan

### Xử lý

- weighted BCE;
- focal loss;
- balanced minibatch;
- PR-AUC;
- Recall@K.

Không dùng Accuracy làm kết luận chính.

---

# 38. Monthly Gates

## Gate M1 — hết tuần 4

Phải có:

- D1 simulation;
- mule labels;
- temporal ground truth;
- leakage audit;
- LightGBM result.

Nếu chưa có:

> không tiếp tục architecture research.

---

## Gate M2 — hết tuần 8

Phải có:

- static GNN;
- temporal GNN;
- STA-GT-inspired baseline;
- multi-scale feature module.

Nếu chưa có:

> bỏ B5/B6 optional.

---

## Gate M3 — hết tuần 12

Phải có:

- role encoder;
- transition encoder;
- EarlyMule-TG;
- preliminary ablation.

Nếu Proposed chưa ổn:

> đóng băng scope và debug.

---

## Gate M4 — hết tuần 16

Phải có:

- frozen test results;
- early detection;
- ablation;
- thesis;
- reproducible code.

---

# 39. Deliverables cuối khóa luận

## DLV1. Source code

Repository hoàn chỉnh.

## DLV2. AMLSim-Mule configuration

- simulation configs;
- typology definitions;
- scripts;
- ground-truth generator.

## DLV3. Processed benchmark

Metadata/data-split files có thể tái tạo từ raw public data.

Không redistrib dataset nếu license không cho phép.

## DLV4. Baseline implementations

- LightGBM/XGBoost;
- GraphSAGE/GAT;
- TGN/TGAT;
- STA-GT-inspired.

## DLV5. EarlyMule-TG

Full proposed model.

## DLV6. Experiment results

- main benchmark;
- ablation;
- label scarcity;
- early detection;
- per-typology.

## DLV7. Thesis

LaTeX + PDF.

## DLV8. Presentation

Khoảng 20–30 phút.

## DLV9. README

Phải cho phép người khác:

1. cài environment;
2. tạo D1;
3. preprocess;
4. train baseline;
5. train Proposed;
6. reproduce main table.

---

# 40. Contribution kỳ vọng của thesis

Không nên viết contribution là:

> “We propose a new GNN architecture.”

Nên phát biểu:

## C1. Mule-centric temporal formulation

Mô hình hóa phát hiện mule như:

\[
Account\text{-}level
+
Point\text{-}in\text{-}time
+
Dynamic\ Transaction\ Graph.
\]

## C2. Multi-scale role dynamics

Biểu diễn hành vi:

\[
Receiver
\rightarrow
Collector
\rightarrow
Relay
\rightarrow
Distributor
\]

tại nhiều thang thời gian.

## C3. Role-transition aware temporal model

Kết hợp:

\[
TemporalGraphEmbedding
+
RoleState
+
RoleTransition.
\]

## C4. Early-detection protocol

Đánh giá không chỉ “đúng/sai” mà còn:

\[
When\ was\ the\ mule\ detected?
\]

bằng:

- TTD;
- Detection Lead Time;
- Recall@K.

## C5. Reproducible mule benchmark

Tạo cấu hình AMLSim-Mule có:

- typology;
- role;
- onset;
- completion;
- temporal ground truth.

---

# 41. Thesis claim cần tránh

Không viết:

> EarlyMule-TG detects real-world bank mule accounts with production-level accuracy.

Vì D1/D2 chủ yếu synthetic.

Nên viết:

> EarlyMule-TG is evaluated on reproducible synthetic AML transaction networks and external financial graph benchmarks to study whether temporal role transitions improve early account-level detection.

Không viết:

> Elliptic++ proves performance in digital banking.

Nên viết:

> Elliptic++ provides cross-domain validation on temporal illicit-address detection.

---

# 42. Suggested thesis structure

## Chapter 1 — Introduction

- digital banking fraud;
- mule accounts;
- motivation;
- research gap;
- research questions;
- contributions.

## Chapter 2 — Background and Related Work

- financial fraud detection;
- AML and mule account detection;
- graph fraud detection;
- temporal GNN;
- recent baselines;
- research gap.

## Chapter 3 — Problem Formulation and Data

- dynamic transaction graph;
- point-in-time prediction;
- mule typologies;
- AMLSim-Mule;
- labels;
- temporal splits;
- leakage controls.

## Chapter 4 — Proposed Method

- temporal graph encoder;
- behavioral features;
- multi-scale role encoder;
- role-transition module;
- fusion;
- training.

## Chapter 5 — Experimental Setup

- datasets;
- baselines;
- metrics;
- label scarcity;
- early detection protocol;
- implementation.

## Chapter 6 — Results and Discussion

- main results;
- ablation;
- early detection;
- label scarcity;
- typology results;
- case study;
- limitations.

## Chapter 7 — Conclusion and Future Work

---

# 43. Research priority order

Thứ tự ưu tiên trong 16 tuần:

\[
\boxed{
Leakage\text{-}safe\ Data
>
Strong\ Baselines
>
Temporal\ GNN
>
Role\ Transition
>
Early\ Detection
>
External\ Validation
}
\]

Không đảo thứ tự.

---

# 44. Quy tắc cắt scope nếu thiếu thời gian

Nếu thiếu thời gian, bỏ theo thứ tự:

1. D3 Elliptic++;
2. FiFrauD implementation;
3. MLaD² contrastive pretraining;
4. explainability;
5. robustness/camouflage test;
6. efficiency benchmark.

**Không được bỏ:**

- D1 AMLSim-Mule;
- point-in-time temporal split;
- LightGBM;
- static GNN;
- temporal GNN;
- STA-GT-inspired baseline;
- role-transition ablation;
- early-detection metrics.

---

# 45. Literature spine

Sinh viên nên đọc theo thứ tự:

## P0 — đọc và ghi chú chi tiết

1. **Tian & Liu, STA-GT, IEEE TII 2024**  
   DOI: <https://doi.org/10.1109/TII.2024.3423447>

2. **Luo et al., MLaD², IEEE TIFS 2024**  
   DOI: <https://doi.org/10.1109/TIFS.2024.3380262>

3. **Khodabandehlou & Hashemi G., FiFrauD, ACM TKDD 2024**  
   DOI: <https://doi.org/10.1145/3641857>

## P1 — đọc để thiết kế mở rộng

4. **Spade+, IEEE TKDE 2024**  
   DOI: <https://doi.org/10.1109/TKDE.2024.3394155>

5. **SEFraud, ACM KDD 2024**  
   DOI: <https://doi.org/10.1145/3637528.3671534>

6. **Grad, ACM Web Conference 2025**  
   DOI: <https://doi.org/10.1145/3696410.3714520>

7. **TF-GNN, Neurocomputing 2026**  
   DOI: <https://doi.org/10.1016/j.neucom.2026.133694>

## Dataset references

8. **IBM AMLSim**  
   <https://github.com/IBM/AMLSim>

9. **IBM AML-Data**  
   <https://github.com/IBM/AML-Data>

10. **Altman et al., Realistic Synthetic Financial Transactions for Anti-Money Laundering Models, NeurIPS 2023**  
    <https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models>

11. **Elliptic++**  
    <https://github.com/git-disl/EllipticPlusPlus>

---

# 46. Definition of Done

Khóa luận được xem là hoàn chỉnh khi có thể trả lời định lượng bốn câu hỏi sau:

### Q1

\[
Temporal\ Graph
>
Static\ Graph?
\]

### Q2

\[
MultiScale\ Role
>
Generic\ Temporal\ Graph?
\]

### Q3

\[
RoleTransition
\Rightarrow
Earlier\ Detection?
\]

### Q4

\[
Performance
\ under
\ Label\ Scarcity?
\]

Nếu trả lời được bốn câu hỏi bằng protocol không leakage, đầy đủ baseline và ablation, thesis đã đạt mục tiêu nghiên cứu.

---

# 47. Định hướng bài báo sau khóa luận

Nếu kết quả tốt, có thể phát triển bài báo theo trục:

> **Early Mule-Account Detection via Multi-Scale Temporal Role Transitions in Dynamic Financial Transaction Graphs**

Paper nên nhấn mạnh:

1. early account-level AML;
2. temporal role transition;
3. leakage-safe evaluation;
4. detection lead time;
5. label scarcity;
6. reproducible mule simulation benchmark.

Không nên mở rộng paper bằng cách ghép quá nhiều module.

Đóng góp cần hội tụ vào một thông điệp:

\[
\boxed{
Mule\ accounts\ are\ easier\ to\ identify\ from\ how\ their\ transaction\ roles\ change\ over\ time,\ not\ only\ from\ what\ their\ static\ graph\ neighborhoods\ look\ like.
}
\]

---

# 48. Kết luận định hướng

Hướng thesis được chốt:

> **Early Mule-Account Detection through Multi-Scale Temporal Role Transitions in Digital Banking Transaction Graphs**

Tên phương pháp:

> **EarlyMule-TG**

Trọng tâm nghiên cứu không phải tạo một GNN lớn hơn, mà là chứng minh ba điểm:

\[
\boxed{
Temporal\ Information
}
\]

có giá trị hơn static graph;

\[
\boxed{
Multi\text{-}Scale\ Role\ Dynamics
}
\]

có giá trị bổ sung so với generic temporal embedding;

và:

\[
\boxed{
Role\ Transition
}
\]

giúp phát hiện mule **sớm hơn**, không chỉ chính xác hơn.

Đây là scope phù hợp cho khóa luận 4 tháng, đồng thời đủ rõ về novelty, baseline, evaluation và khả năng mở rộng thành bài báo nếu kết quả thực nghiệm tốt.
