# Rà soát khoa học và biên tập bản thảo MEPI

Ngày rà soát 27 tháng 9 năm 2026

## Phạm vi và kết luận

Bản thảo được rà soát là `without discussion.docx`, gồm nội dung từ tiêu đề đến hết mục 4.6, sáu hình, sáu bảng kết quả hoặc tổng quan và hai thuật toán. Bản render dùng để kiểm tra có 24 trang. File được gửi không chứa Discussion, Conclusion hoặc danh mục References. Vì vậy, báo cáo này không kết luận rằng các phần đó bị thiếu trong bản hoàn chỉnh, và không xác nhận được toàn bộ ánh xạ [1]–[35] tới tài liệu tham khảo.

Đối chiếu với repo `husphys/trans-core`, nhánh main tại commit `25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8`. Đã kiểm tra mã kiến trúc, hàm loss, tiền xử lý, tạo nhãn, cấu hình, split, bảng kết quả pretraining và depth study, kết quả lựa chọn loss weight, dự đoán test đã lưu và hồ sơ prototype. Không huấn luyện lại hoặc chạy thêm inference trên test. Các thống kê bổ sung trong báo cáo được tính mô tả từ CSV đã công bố, không dùng để chọn lại mô hình.

**Kết luận reviewer:** Bài có dữ liệu thực nghiệm và chuỗi kết quả có thể truy nguyên, nhưng chưa sẵn sàng để gửi theo kỳ vọng của một bài nghiên cứu Q1 về vật liệu hoặc tuổi thọ vật liệu. Vấn đề lớn nhất không phải văn phong mà là tên kiến trúc không khớp thuật toán gốc, công thức loss không khớp mã, độ tin cậy phép đo chưa được xử lý đầy đủ, và thiếu bằng chứng xác định giá trị riêng của từng thành phần MEPI. Định vị có thể bảo vệ hiện nay là nghiên cứu mô hình ước lượng hiệu năng của hai prototype trong miền đo, có regularization dạng Steinmetz và một chỉ số biến đổi từ nhiệt độ.

Q1 là phân hạng tạp chí, không phải một bộ quy tắc dấu câu hoặc một chứng nhận chất lượng áp dụng đồng nhất. Đánh giá dưới đây là nhận định reviewer về khả năng bảo vệ các kết luận, không phải dự báo chắc chắn quyết định biên tập.

Các mã P009, P017… chỉ thứ tự đoạn trong phần thân DOCX, kể cả các đoạn trống hoặc chứa hình. Để tìm trong Word, dùng câu trích kèm theo. Số trang có thể đổi khi mở bằng Word hoặc đổi máy in.

## Các kết quả đã đối chiếu

| Nội dung | Kết quả đối chiếu | Giới hạn diễn giải |
|---|---|---|
| Thiết kế 900 mẫu | Khớp 2 core × 6 voltage labels × 15 frequencies × 5 repeats | Chỉ có hai mẫu vật lý, không phải 900 mẫu vật liệu độc lập |
| 862 mẫu sau QC | Khớp dữ liệu v1.2 đang dùng | Tất cả 862 mẫu là WARNING, không phải sạch mọi cảnh báo |
| Train validation test | Khớp 687, 85, 90 mẫu | Manifest gán 72, 9, 9 groups; train thực tế có 71 groups còn mẫu sau QC |
| Chín features và B(t) 1024 điểm | Khớp mã downstream | Core temperature không nằm trong inputs; ambient temperature có |
| Bảng 3 | Các số khớp CSV backbone comparison | Tên xLSTM và RWKV cần sửa theo implementation |
| Bảng 4 | Các số khớp depth comparison | Caption còn ghi BiGRU; nghiên cứu chỉ có ngân sách 10 epochs |
| Bảng 5 | Khớp 12 cấu hình, minimum 0.202382 tại λ3=1 và λ4=0.05 | Không có λ3=0 hoặc λ4=0 nên không phải ablation chứng minh lợi ích |
| Bảng 6 efficiency | MAE 1.1457 pp, RMSE 1.4834 pp, R² 0.9818 khớp | Sai số so với nhãn được dựng từ phép đo |
| Bảng 6 residual loss | MAE 0.01369 W, RMSE 0.01646 W, R² 0.9706 khớp | Không phải chứng thực phép đo core loss nội tại |
| Bảng 6 LSP | MAE 0.02795, RMSE 0.03637, R² 0.9163, MAPE 4.788% khớp | Khớp một nhãn tất định từ nhiệt độ, không phải nhãn aging |
| Err95 | Khớp 4.4385%, 25.2739%, 13.0185% | Phân vị sai số từng mẫu, không phải khoảng tin cậy 95% |
| Test một lần | Ledger ghi TEST_EVALUATION_COUNT=1 | Đây là bằng chứng lưu trong repo, không phải xác minh độc lập toàn bộ lịch sử vận hành |
| Prototype | Có mã triển khai, ảnh và hồ sơ trình diễn | Chưa có kiểm chứng độc lập về accuracy của live mode hoặc thời gian đáp ứng |

Nguồn repo chính: [kết quả test](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/reports/MEPI_V1_5_FINAL_TEST_RESULTS.json), [dữ liệu QC](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/data/MEPI/v1_2/samples_qc_valid_v1_2.csv), [tổng hợp bằng chứng](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/reports/MEPI_V1_5_MANUSCRIPT_EVIDENCE_SUMMARY.md).

## Các vấn đề có thể làm reviewer bác claim chính

### 1 Tên xLSTM không khớp mô hình được triển khai

Trong `src/mepi_v1/backbones.py`, lớp `XLSTMBlock` gồm LayerNorm, `torch.nn.LSTM`, một linear layer với sigmoid gate, một projection và residual addition. Công thức đúng theo mã là

\[
u=\operatorname{LayerNorm}(x),\qquad
h=\operatorname{LSTM}(u),\qquad
x'=x+\sigma(W_g u+b_g)\odot(W_p h+b_p).
\]

Đây là LSTM chuẩn được bổ sung gated residual connection. Mã này không có exponential gating, scalar memory update kiểu sLSTM hoặc matrix memory kiểu mLSTM của xLSTM do Beck và cộng sự công bố. Việc đặt tên class là XLSTM không đủ để nhận diện nó là kiến trúc xLSTM gốc.

**Hệ quả:** Abstract, mục 3.3.1, 3.5, 4.2, 4.3, các hình 1, 3, 4 và tên dòng trong bảng đều đang gây hiểu nhầm. Không thể dùng bảng 3 để kết luận xLSTM chính thống vượt LSTM, BiGRU hoặc RWKV.

**Sửa giữ nguyên dữ liệu và checkpoint:** gọi mô hình là `LSTM with gated residual connections`, định nghĩa block bằng công thức trên, và ghi một lần rằng artifact lịch sử trong repo dùng nhãn `xLSTM`. Không đổi tên để tạo ấn tượng đây là kiến trúc mới đã được chứng minh độc lập. Nếu muốn tiếp tục claim xLSTM chính thống, phải triển khai đúng rồi thực hiện lại các thí nghiệm chịu ảnh hưởng, đó không còn là sửa văn bản.

Tên RWKV cũng cần thận trọng. `RWKVBlock` hiện dùng trộn token hiện tại với token liền trước theo tỷ lệ 0.5 và các phép gate/projection, không có cơ chế tổng hợp trạng thái WKV theo kiến trúc RWKV gốc. Có thể mô tả là `a simplified RWKV-inspired mixing baseline` và nêu rõ công thức; không trình bày như benchmark đại diện RWKV tiêu chuẩn. LSTM Attention trong mã thực tế dùng BiLSTM rồi self-attention, nên tên mô tả rõ hơn là `BiLSTM with self-attention`.

Nguồn: [backbones.py](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/src/mepi_v1/backbones.py), [xLSTM gốc](https://arxiv.org/abs/2405.04517), [RWKV gốc](https://arxiv.org/abs/2305.13048).

### 2 Equation 11 lớn gấp hai lần electrical loss trong mã

Bài viết định nghĩa electrical loss bằng tổng qua hai targets. Mã `electrical_loss()` dùng `F.mse_loss` và `F.l1_loss` với mean reduction trên tensor có shape N×2. Do đó công thức tương đương với mã phải là

\[
L_{\mathrm{electrical}}=
\frac{1}{2}\sum_{q\in\{\eta,P_{\mathrm{res}}\}}
\left[0.7\operatorname{MSE}(\hat q_z,q_z)+0.3\operatorname{MAE}(\hat q_z,q_z)\right].
\]

Với MSE và MAE của từng target lấy trung bình qua N mẫu, bản thảo hiện thiếu hệ số 1/2. Đây là lỗi tái lập, vì nó thay đổi cân bằng electrical/LSP/physics/NLL và ảnh hưởng ý nghĩa các λ. Sửa Equation 11 và dòng tương ứng trong Algorithm 2 theo mã đã chạy. Không sửa code và không thay bảng kết quả chỉ để khớp công thức viết sai.

Nguồn: [finetune_v1_4.py, electrical_loss và compute_losses](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/src/mepi_v1/finetune_v1_4.py). Đây là hàm loss được workflow v1.5 kế thừa.

### 3 Độ tin cậy phép đo có vấn đề cụ thể cần công bố

Trong chính bảng 862 mẫu đang dùng:

| Trường kiểm tra | Kết quả |
|---|---:|
| qc_status = WARNING | 862/862 |
| TEMPERATURE_CALIBRATION_PENDING | 862/862 |
| VIN_SCOPE1_SCOPE2_MISMATCH | 862/862 |
| VIN_REFERENCE_DRIFT | 766/862 |
| Chênh lệch Vin giữa hai scope | 3.845–7.433%, median 4.655% |
| temperature_calibration_applied | 0 cho toàn bộ mẫu |
| temperature_calibration_status | REFERENCE_BASED_CALIBRATION_NOT_FOUND |

Scope 1 được dùng cho Vin trong tính Pin, Scope 2 được dùng cho Vin của model và tái dựng B(t). Hai scope đo cùng vai trò điện áp sơ cấp nhưng Scope 1 thấp hơn Scope 2 có hệ thống trong dữ liệu. Cần xác định khác biệt đến từ calibration, attenuation, xử lý RMS, thời điểm lấy dữ liệu hay điều kiện thực tế. Chưa đủ bằng chứng để tự chọn một scope là đúng hoặc tự sửa tỷ lệ.

MAE dự đoán 1.1457 pp hoặc 0.01369 W chỉ mô tả mức khớp nhãn. Các chỉ số đó không chứng minh uncertainty của phép đo nhỏ hơn sai số mô hình. Cần measurement uncertainty budget cho điện áp, dòng, phase, điện trở và nhiệt độ, đặc biệt vì residual loss là hiệu giữa các đại lượng công suất.

Từ \(P_{in}=VI\cos\phi\), độ nhạy tương đối với sai số góc nhỏ là \(|\delta P|/P\approx |\tan\phi|\,|\delta\phi|\), với góc theo radian. Vì vậy, cả uncertainty phase và gain đều quan trọng. Đây là phân tích độ nhạy, không phải ước lượng uncertainty thực của thiết bị.

Cụm `corrected measured core temperature` phải nói rõ là sửa ánh xạ core/ambient. Nó không được khiến người đọc hiểu rằng nhiệt độ đã được hiệu chuẩn chuẩn tham chiếu. Cụm `after quality control` vẫn dùng được, nhưng phải nêu rõ tiêu chí loại hard failure và việc giữ các warning.

### 4 LSP không bổ sung bằng chứng tuổi thọ vật liệu

Trong dữ liệu đang dùng, nhiệt độ lõi chỉ từ 26.5 đến 32.25 °C. LSP là biến đổi đơn điệu hoàn toàn xác định từ nhiệt độ này:

\[
\ln LSP=\frac{125000}{R}\left(\frac1T-\frac1{298.15}\right).
\]

Không có aging trajectory, degradation marker, failure criterion hoặc số đo thuộc tính vật liệu trước và sau lão hóa. Vì vậy:

- R² cao của LSP không xác thực cơ chế aging.
- LSP cao không chứng minh lõi có tuổi thọ cao hơn hoặc độ bền vật liệu tốt hơn.
- Không gọi LSP là thermal stress theo nghĩa ứng suất cơ học do giãn nở nhiệt; đó là một temperature-derived index liên hệ hình thức với Arrhenius.
- Chỉ số hiện giảm khi nhiệt độ tăng. Nếu giữ tên thermal-stress proxy, phải nói rõ đây là inverse direction, càng lớn càng thuận lợi về nhiệt độ.

Trong hồ sơ lịch sử của repo, nguồn Arrhenius được dẫn là Bilyaz et al., DOI 10.1016/j.heliyon.2024.e27783. Công trình đó xét suy giảm degree of polymerization của giấy cách điện và nhiệt độ hot spot của winding. Nó không xác lập năng lượng hoạt hóa cho lõi nanocrystalline. Bản DOCX đang gửi không có References, nên cần xác nhận [15] hiện còn là nguồn đó hay đã đổi. Nếu giữ 125 kJ/mol, cần dẫn chính xác nguồn của con số và trình bày đây là giả định quy ước của proxy, không phải tham số đã được đo của vật liệu.

Tại khoảng 300 K, \(d\ln LSP/dT\approx-0.167\ \mathrm{K}^{-1}\). Sai lệch nhiệt độ 0.5 °C có thể làm LSP thay đổi khoảng 8% theo mô hình này. Đây là ví dụ độ nhạy, không phải khẳng định sensor đang sai 0.5 °C. Nó cho thấy vì sao cần kiểm chứng calibration trước khi diễn giải MAPE LSP 4.788% theo nghĩa vật lý.

Một benchmark hữu ích là dự đoán core temperature rồi chuyển sang LSP bằng đúng công thức, so với dự đoán trực tiếp LSP. Hiện chưa có benchmark này, nên không claim LSP head học được aging knowledge. Việc loại Tcore khỏi inputs là đúng để tránh đầu vào chứa đáp án, nhưng không tự tạo novelty về tuổi thọ.

Nguồn nền: [Bilyaz et al. bản PDF tác giả](https://www.fpl.fs.usda.gov/documnts/pdf2024/fpl_2024_bilyaz001.pdf).

### 5 Core loss và residual loss bị dùng lẫn

Abstract và phần mở đầu đã xác định residual loss, nhưng nhiều phần methods và results lại gọi core loss. Mã tính

\[
P_{res}=P_{in}-P_{out}-\left(I_{in}^{2}R_p+I_{out}^{2}R_s\right).
\]

Đây là loss còn lại sau khi trừ ước lượng copper loss. Nó có thể chứa magnetic loss, phần winding loss chưa được mô hình hóa, stray effects và sai số phép đo. Không được đồng nhất với intrinsic magnetic core loss hoặc loss density.

Các vị trí chắc chắn phải sửa nghĩa downstream: P059, P081, P083–P086, P091, P093, P095, P102, P104, P106, P117, P137 ở phần liệt kê downstream targets, P147 và Algorithm 2. Không thay máy móc mọi `core loss`: core loss trong tổng quan tài liệu hoặc source task MagNet vẫn có thể đúng, cần giữ phân biệt.

Nên định nghĩa `P_res` cho văn bản và ghi một lần rằng code lưu dưới tên `P_loss`. Cũng có thể giữ ký hiệu Ploss để giảm sửa, nhưng phải gọi nhất quán là residual loss. Residual loss vật lý khác với residual của dự đoán và khác với Steinmetz residual trong latent module.

Methods cần bổ sung công thức Pin, Pout, Pcu, efficiency và thông số thực dùng: Rload=49.6025 Ω, Rshunt=1.5152 Ω, Rp=0.1012 Ω, Rs=0.1038 Ω. Table 2 có thể ghi `50 Ω nominal, 49.6025 Ω measured`. Cần nói Rp và Rs là số đo nào, ở nhiệt độ nào, có xét AC winding loss hay không. Giá trị giống nhau cho hai core trong CSV cần giải thích hoặc đối chiếu hồ sơ đo, không mặc nhiên coi là lỗi.

### 6 Chưa chứng minh đóng góp riêng của MEPI

Các thí nghiệm hiện có chứng minh rằng một cấu hình cụ thể đạt các chỉ số đã báo cáo. Chúng chưa chứng minh:

- pretraining tốt hơn random initialization trên downstream;
- fusion tốt hơn chỉ waveform hoặc chỉ tabular;
- regularization Steinmetz tốt hơn không regularization;
- NLL/variance branch tốt hơn không branch này;
- một mạng gần 5.84 triệu tham số tốt hơn baseline tabular gọn trên dữ liệu downstream nhỏ.

Backbone comparison trên MagNet không thay thế downstream ablation. Grid λ3 chỉ có 0.1–1.0, không có 0, nên không chứng minh cần physics regularization. Tương tự grid λ4 không có 0. Không nên gọi kiểm tra sensitivity là ablation.

Mức sửa nhẹ bằng văn bản là bỏ claim superiority, enhanced physical consistency hoặc transfer benefit chưa được đối chứng. Nếu muốn đóng góp phương pháp mạnh hơn, phải có đối chứng phù hợp. Các nghiên cứu bổ sung nên xác định trước trên train/validation, giữ nguyên kết quả test đã công bố; không dùng test hiện tại để chọn lại cấu hình rồi trình bày như test chưa từng được xem.

### 7 Hai prototype không đủ để kết luận về lớp vật liệu

Năm repeated measurements tại một điểm không phải năm lõi độc lập. Hai mẫu, một FE và một commercial reference, chỉ hỗ trợ case study của hai thiết bị cụ thể. Group split theo nominal voltage và frequency là lựa chọn tốt để giữ repeats cùng nhóm, nhưng hai core và nhiều session vẫn xuất hiện ở cả train và test. Do đó đây không phải đánh giá trên core mới, batch vật liệu mới hoặc buổi đo độc lập.

Đối với một bài hướng vật liệu, mô tả `iron-based nanocrystalline` và `commercial reference` chưa đủ. Cần composition hoặc product grade, ribbon thickness, annealing condition, trạng thái stress sau cuốn/quấn, coating, packing factor hoặc mật độ, và chứng cứ nhận dạng nanocrystalline liên hệ trực tiếp với mẫu dùng. Nếu dựa vào nghiên cứu [12], phải chỉ rõ mẫu hiện tại thuộc thành phần và quy trình nào của [12]. Không đòi lặp mọi phép XRD/TEM nếu đã có traceability đủ rõ, nhưng không thể chỉ dùng tên vật liệu chung.

Phần results hiện tập trung hầu hết vào ML. Cần một kết quả thực nghiệm trả lời câu hỏi vật liệu, ví dụ so sánh efficiency, residual loss và nhiệt độ của hai prototype theo điều kiện đo, có độ phân tán repeats và giới hạn kiểm soát biến. Nếu không bổ sung, nên thu hẹp định vị thành modeling/instrumentation thay vì chứng minh vật liệu ưu việt.

### 8 Thermal history và ambient temperature có thể gây nhiễu

Nhiệt độ không chỉ phụ thuộc f và Vin tức thời mà còn phụ thuộc thời gian gia nhiệt, trạng thái ban đầu, tản nhiệt và lịch sử sweep. Bản thảo chưa nêu dwell time, tiêu chí ổn định nhiệt, thời gian nghỉ, thứ tự quét hoặc cơ chế randomization.

Trong QC-valid data, median ambient của FE là 27.75 °C, của commercial là 26.25 °C. Median Tcore tương ứng là 28.25 và 27.50 °C, trong khi median temperature rise là 0.75 và 1.25 °C. Các median này gộp điều kiện nên không phải so sánh nhân quả giữa hai vật liệu, nhưng chúng cho thấy xếp hạng theo absolute temperature có thể khác xếp hạng theo self-heating.

Không loại ambient khỏi mô hình chỉ vì nó mang thông tin. Vấn đề là phân biệt thông tin vận hành hợp lệ với session confounding và không diễn giải mô hình học nhiệt độ theo session thành hiểu biết aging. Cần công bố protocol thermal settling hoặc nói rõ dự đoán áp dụng cho điều kiện lấy mẫu của sweep này.

## Physics informed đang có nghĩa gì trong bài

`P_St=k f^alpha B_peak^beta` được fit bằng log-linear least squares trên residual-loss labels của 687 train samples, không dùng core identity. Hệ số lưu trong repo là k=1.638586240032685e-5, alpha=1.4329982981833669, beta=1.8648536845101025, với f theo Hz, B theo T và output theo W.

Đó là empirical reference có dạng Steinmetz, được fit cho hai prototype trong miền đo. Nó không phải một phép đo vật lý độc lập và không phải bộ tham số vật liệu riêng của từng lõi. Cụm `independent physics-based reference` ở P062 cần thay bằng `a fixed Steinmetz-form reference fitted to the training data`.

`L_physics` phạt auxiliary output so với reference. Nó không trực tiếp áp đặt conservation of energy lên outputs cuối. Các prediction heads downstream không bị ràng buộc output efficiency trong [0,100] hoặc residual loss không âm. Vì vậy có thể gọi `Steinmetz-informed regularization`, nhưng không claim physical consistency được bảo đảm.

Fusion chỉ có một tabular token làm key/value. Khi softmax chỉ có một phần tử, attention weight bằng 1. Mô tả hiện tại ở P077 là `global tabular conditioning` đã đúng hơn `token-selective cross-attention`. Không nên nâng phần này thành novelty về lựa chọn attention giữa nhiều modalities.

B(t) được tích phân từ điện áp đầu cực sơ cấp. Phải nêu đây là xấp xỉ dùng geometric cross section Ae=1.217268e-4 m² và không có correction packing factor. Điện áp đầu cực chứa winding/leakage drop nên chưa đồng nhất hoàn toàn với induced emf. Với phân tích tương đối, xấp xỉ có thể chấp nhận nếu công bố rõ. Với kết luận absolute B hoặc cross-material comparison, cần xem sai lệch có đáng kể không. Không thay đổi Ae chỉ vì chưa có packing factor.

## Novelty có thể bảo vệ

| Thành phần | Có thể gọi là novelty không | Claim phù hợp |
|---|---|---|
| Hai prototype và dataset paired | Đóng góp thực nghiệm có điều kiện | Một dataset thiết bị trong miền vận hành xác định |
| Chuyển representation từ MagNet sang ba targets downstream | Điểm khác biệt ứng dụng đáng khảo sát | Áp dụng transfer initialization cho bài toán downstream cụ thể |
| xLSTM | Không theo tên hiện tại | Mô tả đúng custom residual-gated LSTM |
| Tách Arrhenius target khỏi Steinmetz regularizer | Quyết định thiết kế hợp lý | Cấu trúc phương pháp rõ ràng; không tự là phát minh vật lý |
| Loại target leakage và tách train validation test | Yêu cầu phương pháp cơ bản | Bằng chứng tính đúng của thí nghiệm |
| LSP từ nhiệt độ | Không phải novelty tuổi thọ | Chỉ số quy ước để so sánh trạng thái nhiệt |
| GUI và live inference | Đóng góp triển khai | Prototype hoạt động theo phạm vi bằng chứng đã có |

MagLearn đã nghiên cứu transfer/few-shot learning cho magnetic loss từ waveform. Vì thế không thể claim waveform transfer learning nói chung là mới. Novelty cần đặt ở câu hỏi downstream, loại phép đo có thể thay thế, hoặc giá trị thực nghiệm trên prototype. Một bài không trở nên mới chỉ vì chưa có bài trước dùng đúng cả ba tên outputs của mình.

Nguồn: [MagLearn, University of Bristol](https://research-information.bris.ac.uk/en/publications/maglearn-data-driven-machine-learning-framework-with-transfer-and/).

Định vị đề xuất, chỉ dùng sau khi sửa tên mô hình và các lỗi phương pháp:

> This study evaluates a multimodal surrogate for estimating transformer efficiency, residual loss, and a temperature-derived index from voltage waveforms and operating measurements. The evaluation uses two toroidal prototypes and separates operating-condition groups during model development and testing.

Điểm ứng dụng có thể mạnh hơn hiện tại: lúc thu dataset cần current/phase/temperature instrumentation để tạo reference labels, trong khi inference prototype chỉ cần hai voltage channels và ambient temperature. Đây là hướng sensor reduction hoặc virtual sensing. Muốn claim có lợi thực tế phải lượng hóa thiết bị được giảm, accuracy, latency và miền áp dụng, không chỉ chụp GUI.

## Những mâu thuẫn phương pháp và trình bày cần sửa

| Vị trí | Vấn đề | Sửa tối thiểu |
|---|---|---|
| Table 4 caption, P140 | Ghi BiGRU và MEPI performance | `Effect of backbone depth on MagNet validation performance`, sau khi định danh đúng backbone |
| P142 | 5,969,930 bị gán cho eight-layer backbone | Đây là tổng model pretraining; downstream là 5,840,389 parameters; không gọi là số riêng backbone |
| Tables 3 và 4 | Không giải thích 20 epochs so với 10 epochs | Nêu rõ ngân sách; depth-4 0.015028 và 0.026314 không phải cùng một run |
| P141–142 | Dễ khiến hiểu depth 8 là tối ưu nói chung | Chỉ là best trong 5 depths dưới ngân sách 10 epochs; depth 8 best ở epoch cuối |
| P059 | Nói tất cả normalization fit từ train mà không phân biệt source/downstream | Waveform scaler kế thừa MagNet train; tabular và target scalers fit downstream train |
| Methods MagNet | Thiếu số mẫu, materials, source split, target transform | Bổ sung 186,757 mẫu, 10 materials, split 149,405/18,676/18,676 và standardized log1p target |
| Algorithm 1 | Chỉ mô tả một temporary head | Code dùng 10 material-specific regression heads, routing theo material metadata |
| Algorithm 1 | `return` nằm trước khi hoàn tất vòng backbone; không có argmin cuối | Tách lưu best checkpoint từng candidate và chọn candidate theo validation MAE |
| Algorithm 1 | `learning rate schedule` nhưng code common run không có scheduler | Mô tả learning rate và optimizer thực dùng |
| Algorithm 2 | Init model ở ngoài vòng λ | Di chuyển reset seed, load representation, init modules và optimizer vào mỗi candidate; code đã làm đúng |
| Algorithm 2 | Thiếu cập nhật winner theo Sval; dòng cuối lặp 21 | Thêm argmin giữa checkpoints, sửa thứ tự và điều kiện end if |
| Equation 1 | `log` không nêu cơ số | Dùng `ln`, nhiệt độ tuyệt đối theo kelvin |
| Equation 5 | Bản render hiển thị `ln` thay cho LayerNorm | Dùng toán tử `LayerNorm` hoặc upright `LN`, tránh nhầm natural logarithm |
| Equation 12 | μLSP thiếu chỉ số z so với phương trình NLL | Thống nhất μLSP,z ở tất cả normalized losses |
| Equation 19 và 20 | ε chỉ gọi nhỏ, chưa cho số | Ghi ε=1e-12 theo helper metrics đang dùng |
| P125 | `Tn addition` | `In addition` |
| P101 | `Where` đầu phần giải thích | `where` nếu nối với công thức |
| P147 | `0.01140 for core loss` thiếu W và sai tên | `0.01140 W for residual loss` |
| Table 4 | Peak GPU memory không có đơn vị | Ghi GiB theo CSV nguồn |
| P148 | `epoch 91 using zero based indexing` | `the 92nd epoch`, nêu zero-based index trong supplement nếu cần |
| All equations | Ký tự # xuất hiện cạnh số equation trong bản render | Bỏ # literal và căn số (1), (2)… bên phải |
| Algorithms | Nhiều dòng bị dính công thức và số bước | Tách dòng, tránh full justification làm khoảng trắng giãn bất thường |

## Các câu nên xóa hoặc viết lại

Các câu dưới đây không bị đánh giá là “do AI viết”. Không thể xác định nguồn tác giả chỉ từ văn phong. Vấn đề là nội dung lặp, khái quát quá rộng, tự thuật quá trình viết hoặc không cung cấp kết quả có thể kiểm tra.

| Mã và vị trí | Câu hoặc cụm nguyên văn để tìm | Hành động và lý do |
|---|---|---|
| E01 P012 | “Energy dissipation remains an important consideration in transformer operation” | Bỏ mở đầu hiển nhiên. Mở trực tiếp bằng vấn đề loss trong miền thiết bị đang nghiên cứu |
| E02 P012 | “Reducing magnetic losses can therefore contribute to improved operating efficiency and lower life-cycle emissions” | Cắt nếu không có phân tích lifecycle; đã suy ra từ hai câu trước |
| E03 P013 | “Advanced soft magnetic materials have therefore been investigated to overcome these limitations” | Gộp với câu sau và nêu vật liệu cụ thể |
| E04 P013 | “These characteristics make them promising candidates” | Bỏ lời đánh giá chung nếu không nối với điều kiện f/B cụ thể |
| E05 P014 | “the coupled effects ... remain insufficiently characterized” | Giới hạn rõ nghiên cứu nào và khoảng điều kiện nào, nếu không đây là gap claim không kiểm chứng |
| E06 P015 | “These relations address different physical aspects of transformer operation and should therefore be used according to their respective roles.” | Xóa, câu sau đã nói chính xác vai trò |
| E07 P016 | “Recent advances in artificial intelligence have enabled complex nonlinear relationships to be learned directly from experimental measurements” | Xóa, là mở đầu AI chung chung |
| E08 P016 | “Such approaches are particularly attractive when waveform-level information must be combined with heterogeneous electrical, magnetic, and thermal variables.” | Xóa hoặc thay bằng vấn đề input hiện có và sensor nào thiếu lúc inference |
| E09 P016 | “Despite recent progress, several challenges remain” | Xóa, trình bày thẳng challenge có bằng chứng |
| E10 P017 | “MEPI, meaning Multimodal Enhanced Physics Informed learning” | Định nghĩa tên đúng một lần; từ Enhanced cần đối chứng hoặc bỏ trong tên mô tả |
| E11 P019 | “a leakage controlled multimodal learning framework” | Không đặt leakage control như novelty chính; chuyển sang Data partitioning |
| E12 P020 | “The framework explicitly separates target construction from physics-based model regularization.” | Có thể giữ một lần trong Methods, bỏ như contribution độc lập nếu không có kiểm chứng lợi ích |
| E13 P021 | “The frozen MEPI model is evaluated once on the final test set” | Chuyển vào evaluation protocol; đây là cách làm đúng, không phải discovery |
| E14 P023 | “The remainder of this paper is organized as follows.” | Có thể xóa toàn đoạn nếu journal không yêu cầu roadmap |
| E15 P025 | “A device level assessment must connect these effects with electrical performance and thermal response.” | Xóa, nội dung chung và lặp Introduction |
| E16 P027 | “These studies support the use of learned representations and multimodal fusion” | Cắt ví dụ gas turbine/motor nếu không dẫn trực tiếp đến phương pháp cụ thể |
| E17 P028 | “They also do not establish a universally superior sequence backbone.” | Xóa. Đây không phải thiếu sót của các nghiên cứu đó |
| E18 P029 | “This distinction avoids treating target construction and model regularization as the same physical constraint.” | Xóa nếu đã định nghĩa rõ ở Methods |
| E19 P032 | “The unresolved problem is not the absence of any one component. Rather, it is the lack of...” | Bỏ cấu trúc tu từ; viết vấn đề đo/dự đoán cụ thể |
| E20 P038 | “The implemented downstream architecture therefore combines transferred representation components with task-specific modules trained for transformer-level assessment.” | Xóa, lặp chính câu ngay trước |
| E21 P039 | “After validation-based model development is completed, the selected MEPI configuration and checkpoint are frozen.” | Giữ chi tiết tại 3.5/4.5, bỏ đoạn lặp tại 3.1 |
| E22 P045 | Equation 2 chỉ là exp(log LSP) | Có thể gộp Equation 1–2 thành một định nghĩa exponential; log-space computation mô tả một câu |
| E23 P061 | “In this section, the architecture of MEPI is presented.” | Xóa hoàn toàn |
| E24 P063 | “This separation maintains a clear distinction between physics-based target construction and physics-guided representation learning.” | Xóa, đã nhắc nhiều lần |
| E25 P066 | “These stages share a common learned representation while retaining distinct roles for data-driven feature extraction, physics-informed refinement, and task-specific estimation.” | Xóa, mô tả vòng lại hình mà không có thông tin mới |
| E26 P066 | “The following subsections present the formulation and implementation of each stage in detail.” | Xóa hoàn toàn |
| E27 P079 | “Transfer is restricted to the waveform encoder and xLSTM backbone.” | Giữ thông tin một lần ở training, bỏ các lần lặp ở 3.1/3.3.1 nếu không cần |
| E28 P084 | “defines the Steinmetz residual, as defined in Equation 8” | `The Steinmetz residual is defined in Equation (8).` |
| E29 P089 | “The residual formulation retains the multimodal representation while introducing information associated with disagreement...” | Xóa, Equation 9 đã cho đúng nội dung này |
| E30 P091 | “The resulting physics informed representation is subsequently used for joint prediction...” | Xóa, lặp câu mở 3.3.3 |
| E31 P093 | Câu giải thích MSE là mean squared errors và MAE là mean absolute errors | Rút gọn, không cần định nghĩa cơ bản rồi lại giải thích ở 4.1 |
| E32 P102 | “The resulting MEPI framework jointly predicts...” | Xóa câu tổng kết lặp |
| E33 P107 | “Experimental verification is required before a screened condition can be considered for operating point optimization.” | Giữ ý giới hạn một lần, gộp với câu không claim optimum |
| E34 P110 | “rather than the logged training mini batch error” | Bỏ so sánh không cần thiết; chỉ nêu validation MAE là criterion |
| E35 P115 | “This procedure separates representation selection from downstream task optimization...” | Xóa, các bước đã trình bày đầy đủ |
| E36 P117 | Toàn đoạn roadmap của Results | Rút xuống một câu, hoặc bỏ; hiện còn hứa computational characteristics nhưng phần sau chưa có kết quả tương ứng |
| E37 P119 | “complementary regression metrics that characterize average prediction error, sensitivity to relatively large deviations, goodness of fit, and upper-tail error behavior” | `Prediction errors were evaluated using MAE, RMSE, R², and Err95.` |
| E38 P130 | Lặp lifetime, RUL, time to failure, measured degradation | Giữ một câu ngắn về constructed labels; định nghĩa giới hạn đầy đủ chỉ cần một lần |
| E39 P144 | “all of which completed without numerical failure and were eligible for model selection” | Bỏ khỏi results chính hoặc chuyển supplement; hoàn thành chạy không phải kết quả khoa học nổi bật |
| E40 P144 | Giải thích Sval lặp lại mục 3.5 | Xóa phần lặp, dẫn Equation (15) |
| E41 P148 | “completed the prescribed 100 epoch training horizon” | `Training reached the maximum of 100 epochs without triggering early stopping.` Vì 100 là maximum, không phải bắt buộc |
| E42 P150–152 | Lặp toàn bộ Table 6 bằng văn xuôi | Giữ một kết quả chính và phân tích bias/điểm yếu, không đọc lại mọi ô |
| E43 P155 | “Together, the six panels complement the aggregate performance metrics...” | Xóa, thay bằng nhận xét về residual loss lệch dương |
| E44 P157 | “To demonstrate the practical implementation of the developed framework” | `The frozen model was deployed in a monitoring prototype...` |
| E45 P160 | Ba câu liên tiếp “Figure 6(a) shows… Figure 6(b) presents… Figure 6(c) shows…” | Chuyển mô tả panels vào caption, phần text nói chức năng và kết quả kiểm chứng |
| E46 P161 | “without additional training or model adaptation” | Giữ một lần nếu cần phân biệt inference với online learning; tránh lặp frozen model khắp bài |

Các cảnh báo phạm vi không nên bị xóa sạch để làm bài mạnh hơn. Giữ một định nghĩa rõ của LSP, một đoạn limitations đầy đủ và một câu giới hạn screening. Mục tiêu là loại lặp, không che hạn chế.

## Các chỗ thay dấu câu cụ thể

Không có quy tắc Q1 cấm dấu gạch nối, chấm phẩy hoặc hai chấm. Dấu gạch nối trong compound modifier thường đúng ngữ pháp. Khi muốn giảm dấu, cần viết lại cấu trúc, không chỉ xóa ký tự.

| Vị trí | Bản hiện tại | Cách sửa |
|---|---|---|
| Abstract P009 | “... R² of 0.9818; residual loss achieved ...; and LSP achieved ...” | Tách thành 2–3 câu hoặc chỉ nêu MAE ở abstract, giữ các metric còn lại trong Table 6 |
| P017 | “Physical knowledge is introduced through two distinct mechanisms: the Arrhenius relation ...” | `The Arrhenius relation defines LSP, and a Steinmetz-form reference regularizes an auxiliary loss estimate.` |
| P017 | “The main contributions of this study are as follows:” | Nếu tiếp tục list thì dấu hai chấm đúng; nếu bỏ list, dùng `The study provides...` rồi nêu contribution đã được chứng minh |
| P038 | “The complete downstream architecture is not pretrained on MagNet; the transformer-specific ...” | `Only the waveform encoder and sequence backbone are transferred from MagNet. The remaining modules are initialized for the transformer task.` |
| Fig. 2 caption P051 | “Experimental measurement setup; (a) schematic diagram and (b) practical test bench” | `Experimental setup showing (a) the measurement circuit and (b) the laboratory test bench.` |
| P066 | “follows three principal stages: multimodal representation and fusion, ...” | `The computation includes multimodal fusion, Steinmetz-informed refinement, and multitask prediction, as shown in Fig. 3.` Hoặc bỏ vì lặp |
| P059 | “operating-condition group level” | `by groups defined by nominal voltage and frequency` |
| P068 | “task-specific latent representation” | `a latent representation for the transformer task` |
| P079 | “eight-block xLSTM backbone” | `a backbone with eight gated residual LSTM blocks` sau khi sửa định danh |
| P081 | “inference computable physical reference” | `a reference computed from the available inputs during inference` |
| P017/P058 | “target-derived quantities” | `quantities derived from the prediction targets` |
| P028 | “waveform-based learning” | `learning from waveforms` |
| P019 | “validation-selected waveform representation” | `a waveform representation selected using validation data` |
| P021/P157 | “frequency-response screening” | `screening across the measured excitation frequencies` |
| P152 | “95th-percentile relative errors” | `the 95th percentile of the relative errors` |
| Fig. 5 caption | “Final-test prediction and residual diagnostics” | `Prediction and residual diagnostics on the final test set` |

`include` chỉ dùng với danh từ hoặc cụm danh từ, ví dụ `The predictive inputs include frequency, voltage, and ambient temperature.` Khi liệt kê đủ đúng chín features, `comprise` hoặc `consist of` rõ hơn. Không dùng `include` để nối tùy ý hai mệnh đề độc lập.

Không đổi dấu trừ trong công thức, dấu âm trong exponent, dấu gạch biểu thị khoảng refs [18–21], hoặc panel range (a–c) thành dấu phẩy. Những ký hiệu này mang nghĩa kỹ thuật khác với punctuation tu từ.

## Cách dẫn hình bảng và công thức

Theo phong cách tác giả mong muốn, viết câu có nội dung rồi kết thúc bằng dẫn chiếu. Không bắt buộc mọi câu cùng một mẫu. `Table` hay `Tab.` nên tuân theo journal; nếu giữ `Tab.` thì nhất quán trong body text, còn caption có thể vẫn dùng `Table` theo template.

| Vị trí | Câu thay thế đề xuất |
|---|---|
| P030, Table 1 | `The scope and prediction targets of the reviewed studies are compared in Tab. 1.` |
| P052, Table 2 | `The operating grid comprises six nominal voltage levels and 15 excitation frequencies, as shown in Tab. 2.` |
| P133, Table 3 | `The gated residual LSTM achieved the lowest validation MAE among the evaluated implementations, as shown in Tab. 3.` |
| P139, Table 4 | `Eight blocks gave the lowest validation MAE within the 10-epoch depth study, as shown in Tab. 4.` |
| P144, Table 5 | `The minimum validation task score was obtained with λ3=1.0 and λ4=0.05, as shown in Tab. 5.` |
| P150, Table 6 | `The final test errors for efficiency, residual loss, and LSP are reported in Tab. 6.` |
| P152/P155, Fig. 5 | `Residual loss was overpredicted for 80 of the 90 test measurements, as shown in Fig. 5(e).` Số 80/90 được tính từ CSV đã lưu |
| P157, Fig. 6 | `The frozen model was implemented in a prototype with offline screening and live monitoring modes, as shown in Fig. 6.` |
| Equation 1 | `The temperature-derived index is defined in Equation (1).` |
| Equation 8 | `The Steinmetz residual is defined in Equation (8).` |
| Equation 10 | `The regularization loss is defined in Equation (10).` |
| Equation 11 | `The electrical prediction loss is defined in Equation (11).` |
| Equation 15 | `The score used to compare the loss-weight configurations is defined in Equation (15).` |

Xóa mẫu lặp `The comparative results are summarized, as shown in Tab. ...` tại P133 và P139, `The complete acquisition configuration ... are summarized, as shown ...` tại P052, và mẫu tương tự P030/P144. `summarized` và `shown` ở đây làm cùng một việc.

`is defined in Equation` phù hợp cho một định nghĩa. Đối với đại lượng thực sự được tính bằng quan hệ vật lý, `is calculated using Equation (3)` chính xác hơn. Không cần ép mọi phép biến đổi thành một định nghĩa mới.

## Nhận xét riêng từng hình

**Fig. 1:** Hình hiện cho hai dataset đi vào cùng một encoder và một khối MEPI, dễ tạo cảm giác toàn bộ MEPI đã được pretrained trên MagNet. Nên chia source training và downstream training, thể hiện chỉ waveform encoder và backbone được transfer. Bổ sung temporary MagNet head và nêu modules mới. Sửa nhãn xLSTM theo implementation.

**Fig. 2:** Sơ đồ tối giản chưa thể hiện đủ Scope 1/Scope 2, kênh phase và vị trí hai sensor. Nên có labels CH1/CH2, node trước/sau shunt, đo điện áp tải và hai vị trí nhiệt độ. Cần phân biệt hai phép đo đồng bộ trong từng scope với đồng bộ giữa hai scope. Mô tả hiện tại không nên khiến hiểu cả hệ thống được trigger đồng bộ nếu chưa có bằng chứng.

**Fig. 3:** Khối Steinmetz reference chưa có đường input rõ từ f và Bpeak. Hình chưa thể hiện phép trừ tạo rSt, projection và residual addition vào s. Đây mới là cấu trúc tác giả dùng để claim physics guidance, nên cần minh họa rõ thay vì một hộp tổng quát.

**Fig. 4:** Hiện lặp phần trái Fig. 3 và không làm rõ CNN/FFT, token shapes hoặc one-token conditioning. Có thể gộp vào Fig. 3, hoặc thay bằng chi tiết waveform encoder 1024→256 tokens và tabular token để hình có giá trị mới.

**Fig. 5:** Hình có giá trị vì thể hiện residuals. Text hiện chỉ kể panel, bỏ lỡ kết quả chính: residual loss có bias dương. Từ CSV, mean residual loss là +0.01247 W và 80/90 dự đoán cao hơn reference. Efficiency có mean residual −0.4879 pp. Đây là thống kê mô tả, chưa phải significance test. Có thể phân biệt hai core bằng màu hoặc marker để tránh aggregate metrics che khác biệt.

**Fig. 6:** Ảnh prototype có giá trị minh chứng triển khai. Screenshots b/c rất nhỏ trong bản render, cần crop hoặc tăng kích thước nếu muốn đọc thông số. Không dùng GUI snapshot làm bằng chứng predictive accuracy. Bổ sung số rows demo 150→146 sau QC, nominal 3.9 V và 15 frequencies trong text nếu mô tả đúng dataset được minh họa. Nếu không có latency benchmark thì bỏ từ `rapid` trong claim cuối abstract.

## Tăng sức thuyết phục bằng kết quả đã có

Không cần thay checkpoint để làm các bước mô tả sau:

1. Công bố số cảnh báo QC, tiêu chí loại mẫu và hạn chế calibration.
2. Thêm kết quả thực nghiệm theo core và điều kiện thay vì chỉ bảng accuracy.
3. Trình bày signed residual và phân tích bias từ file prediction đã lưu.
4. Tách mean errors theo core, ghi rõ phân tích hậu nghiệm mô tả.
5. Phân biệt source model parameter count với downstream model parameter count.
6. Nêu rõ source dataset, scaler, material-routed heads và ngân sách epochs.
7. Cung cấp raw-data access route nếu muốn tuyên bố tái lập từ waveform gốc.

Thống kê hậu nghiệm từ test predictions đã lưu:

| Core | n | Efficiency MAE pp | Residual loss MAE W | LSP MAE |
|---|---:|---:|---:|---:|
| FE | 45 | 1.28037 | 0.0120085 | 0.0251535 |
| Commercial | 45 | 1.01096 | 0.0153766 | 0.0307479 |

Không coi 90 samples là 90 quan sát độc lập hoàn toàn. Nếu trình bày uncertainty của metric, resampling cần giữ cấu trúc operating-condition group, thay vì bootstrap từng row mà bỏ phụ thuộc repeats. Chỉ có 9 test groups nên bất định của generalization cần diễn giải thận trọng. Không gọi Err95 là confidence interval.

## Khả năng tái lập và dữ liệu công khai

Repo có derived model inputs, B1024 arrays, split và checkpoints, giúp kiểm tra pipeline cuối. Tuy nhiên `DATA.md` xác nhận raw acquisition archive gồm 4,530 files và MagNet source HDF5 không được lưu trong repo. Hash inventories không thay thế file gốc. Vì thế không nên viết rằng toàn bộ raw experimental data đã được công bố trên GitHub. Nên ghi rõ derived data công khai, raw files lưu ở đâu hoặc có cơ chế truy cập nào.

Hồ sơ cũ có trạng thái TRAIN_READY=FALSE và LSP chưa định nghĩa. Các trạng thái đó đã được v1.2/v1.5 thay thế và không được dùng để kết luận pipeline hiện tại chưa train. Khi trích dẫn cần nêu version. Ngược lại, các vấn đề còn hiện diện trong current CSV, như calibration pending, vẫn phải xử lý dù báo cáo readiness đã PASS.

Nguồn: [DATA.md](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/DATA.md), [physical demonstration evidence](https://github.com/husphys/trans-core/blob/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8/evidence/physical_demo/README.md).

## Thứ tự sửa ưu tiên

**Trước hết, sửa tính đúng:** định danh backbone, Equation 11, residual-loss terminology, Algorithm 1/2, caption Table 4 và normalization provenance. Những việc này có thể thực hiện mà không huấn luyện lại.

**Tiếp theo, xử lý độ tin cậy thực nghiệm:** giải thích scope mismatch, calibration nhiệt độ, copper-loss approximation, thermal settling và nhận dạng vật liệu. Đây là phần không thể chữa chỉ bằng cách viết thận trọng hơn nếu muốn duy trì claim vật liệu mạnh.

**Sau đó, chốt mức claim:** nếu giữ dữ liệu và thí nghiệm hiện tại, định vị case study về surrogate prediction và prototype. Nếu muốn claim lợi ích riêng của transfer, physics regularization hoặc multimodal fusion, bổ sung đối chứng đã xác định trước. Nếu muốn claim tuổi thọ vật liệu, cần thiết kế aging evidence phù hợp, không thể suy ra từ LSP hiện tại.

**Cuối cùng, cắt văn phong:** dùng danh sách E01–E46, rút các đoạn lặp về transfer, frozen checkpoint, leakage và LSP limitations; thay mô tả hình/bảng bằng nhận xét kết quả. Sau khi sửa công thức phải cập nhật lại mọi cross-reference.

## Nguồn kiểm tra chính

- Repository snapshot: https://github.com/husphys/trans-core/tree/25aa4b9bf79c1dd8b30c3fbb3db731885b2b68a8
- `src/mepi_v1/backbones.py`, `models.py`, `finetune_v1_4.py`, `loss_weight_sensitivity_v1_5.py`, `build_finetune_dataset.py`, `pretrain.py`.
- `data/MEPI/v1_2/samples_qc_valid_v1_2.csv`, `split_manifest_v1_2.csv`, `train_normalization_v1_2.json`.
- `reports/pretraining_backbone_comparison.csv`, `experiments/xlstm_depth_v1/depth_comparison.csv`.
- `reports/MEPI_V1_5_FINAL_TEST_RESULTS.json`, `MEPI_V1_5_POSTHOC_TEST_PREDICTIONS.csv`, `MEPI_V1_5_POSTHOC_TEST_METRICS.json`.
- `artifacts/finetune_v1_4/steinmetz_prior_train_v1_4.json`.
- Beck et al. xLSTM: https://arxiv.org/abs/2405.04517
- Peng et al. RWKV: https://arxiv.org/abs/2305.13048
- Zhang et al. MagLearn: https://doi.org/10.1109/ECCEEurope62508.2024.10751860
- Bilyaz et al. Modeling the impact of high thermal conductivity paper on the performance and life of power transformers: https://doi.org/10.1016/j.heliyon.2024.e27783

Phạm vi tìm tài liệu ngoài repo ở đây tập trung vào định danh kiến trúc và nền tảng Arrhenius/transfer. Đây không phải systematic literature review đủ để chứng nhận claim “first” hoặc “unique”.
