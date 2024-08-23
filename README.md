# Dự Báo Mực Nước Trên Sông Nhật Lệ Bằng Mạng LSTM Kết Hợp Cơ Chế Attention
## Giới Thiệu
Dự án này tập trung vào việc phát triển một mô hình dự báo mực nước trên sông Nhật Lệ sử dụng mạng LSTM (Long Short-Term Memory) kết hợp với cơ chế Attention. Mô hình này có khả năng dự báo mực nước sau 3, 6, 12, và 24 giờ, giúp đưa ra các cảnh báo sớm và hỗ trợ trong việc quản lý nguồn nước, phòng chống thiên tai.

## 🧠 Mô Hình LSTM Và Attention
### Mạng LSTM (Long Short-Term Memory)
LSTM là một loại mạng nơ-ron hồi quy (RNN) đặc biệt, được thiết kế để giải quyết vấn đề về sự biến mất hoặc bùng nổ của gradient khi huấn luyện các mô hình RNN trên các chuỗi dữ liệu dài. LSTM có khả năng lưu trữ thông tin quan trọng trong một khoảng thời gian dài và loại bỏ thông tin không cần thiết thông qua các cổng điều khiển (gates) như cổng quên (forget gate), cổng nhập (input gate), và cổng xuất (output gate).

Trong bối cảnh dự báo mực nước, LSTM giúp nắm bắt và học các mẫu tuần hoàn hoặc các biến động phức tạp trong chuỗi thời gian, từ đó cải thiện khả năng dự báo mực nước tại các khoảng thời gian tương lai.

### Cơ Chế Attention
Cơ chế Attention được thiết kế để giúp mô hình tập trung vào các phần quan trọng của chuỗi dữ liệu đầu vào trong khi thực hiện dự báo. Điều này đặc biệt hữu ích khi chuỗi dữ liệu có độ dài lớn và chỉ một số phần của nó là quan trọng đối với dự báo tại một thời điểm cụ thể.

Trong mô hình kết hợp LSTM-Attention của chúng tôi, cơ chế Attention sẽ giúp mô hình "chú ý" đến các thời điểm quan trọng trong chuỗi dữ liệu mực nước, giúp cải thiện độ chính xác của dự báo bằng cách tăng cường sự tập trung vào các phần quan trọng này.

### Lợi Ích Của Việc Kết Hợp LSTM Với Attention
Khả Năng Nắm Bắt Mối Quan Hệ Dài Hạn: LSTM có khả năng xử lý thông tin trên chuỗi thời gian dài, nhưng khi kết hợp với Attention, mô hình không chỉ nắm bắt được thông tin dài hạn mà còn tập trung vào các thời điểm quan trọng nhất.

Cải Thiện Độ Chính Xác Dự Báo: Attention cho phép mô hình tập trung vào các phần dữ liệu có ý nghĩa nhất, giúp tăng cường độ chính xác của dự báo, đặc biệt là trong các hệ thống phức tạp như mực nước sông.

## Đóng Góp
Người hướng dẫn: TS.Nguyễn Thị Kim Ngân
Nguyễn Đức Tuấn
Nguyễn Trung Tuyến
Nguyễn Anh Tuấn
Bùi Tuấn Minh

## Liên Hệ
Nếu có bất kỳ câu hỏi hoặc thắc mắc nào, vui lòng liên hệ qua email 0310ngtuan@gmail.com

