// GenData.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

    cv::Mat imgTrainingNumbers;         // ảnh đầu vào
    cv::Mat imgGrayscale;               // 
    cv::Mat imgBlurred;                 // khai báo các loại ảnh biến đổi
    cv::Mat imgThresh;                  //
    cv::Mat imgThreshCopy;              //

    std::vector<std::vector<cv::Point> > ptContours;        // khai báo contours vector
    std::vector<cv::Vec4i> v4iHierarchy;                    // khai báo contours hierarchy

    cv::Mat matClassificationInts;      // vector nhãn

                                        // đây là traning image, do kiểu dữ liệu mà KNN yêu cầu, chúng ta cần khai báo dạng single Mat,
                                        // sau đó thêm vào vector
    cv::Mat matTrainingImagesAsFlattenedFloats;

    // các kí tự hợp lệ
    std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };

    imgTrainingNumbers = cv::imread("training_nums.png");          // đọc ảnh training

    if (imgTrainingNumbers.empty()) {                               // nếu mở file thất bại
        std::cout << "error: image not read from file\n\n";         // thông báo lỗi
        return(0);                                                  // và thoát khỏi chương trình
    }

    cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);        // biến đổi về grayscale

    cv::GaussianBlur(imgGrayscale,              // ảnh đầu vào
        imgBlurred,                             // ảnh đầu ra
        cv::Size(5, 5),                         // khai báo ksich thước cửa sổ trượt
        0);                                     // giá trị sigma, xác định mức độ làm mờ ảnh, giá trị 0 làm hàm tự xác định giá trị

                                                // biến đổi ảnh từ grayscale sang đen trắng
    cv::adaptiveThreshold(imgBlurred,           // ảnh đầu vào
        imgThresh,                              // ảnh đầu ra
        255,                                    // biến đổi các pixel đạt ngưỡng thành hết màu trắng
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // dùng gaussian hơn là mean, có vẻ đưa ra kết quả tốt hơn
        cv::THRESH_BINARY_INV,                  // đảo ngược lại màu chữ thành trắng, nền thành đen
        11,                                     // kích cỡ pixel lân cận sử dụng tính toán giá trị ngưỡng
        2);                                     // trừ một giá trị hằng số từ giá trị trung bình hoặc trung bình trọng số

    cv::imshow("imgThresh", imgThresh);         // hiển thị ảnh ngưỡng để tham chiếu

    imgThreshCopy = imgThresh.clone();          // tạo bản sao của ảnh ngưỡng, điều này là cần thiết vì hàm finContour có thể thay đổi ảnh

    cv::findContours(imgThreshCopy,             // ảnh đầu vào, hãy chắc chắn rằng đây là ảnh copy
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // chỉ trả về contours khít nhất
        cv::CHAIN_APPROX_SIMPLE);               // co contour lại

    for (int i = 0; i < ptContours.size(); i++) {                           // với mỗi contour
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                // nếu contour đủ lớn
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // tạo hình chữ nhật bao quanh

            cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      // vẽ hình chữ nhật quan contour

            cv::Mat matROI = imgThresh(boundingRect);           // tạo ảnh ROI của khối bao đóng

            cv::Mat matROIResized;
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, tăng tính nhất quán cho việc nhận diện và lưu trữ
            cv::imshow("matROI", matROI);                               // hiển thị ảnh ROI để so sánh
            cv::imshow("matROIResized", matROIResized);                 // hiển thị ảnh ROI được thay đổi kích cỡ để so sánh
            cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       // hiển thị ảnh số training, vẽ hình chữ nhật màu đỏ quanh số đó

            int intChar = cv::waitKey(0);           // đọc phím bấm

            if (intChar == 27) {        // nếu phím Esc được bấm
                return(0);              // thì thoát chương trình
            }
            else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {     // kiểm tra xem kí tự nhập vào có hợp lệ không

                matClassificationInts.push_back(intChar);       // thêm kí tự vừa nhập vào list

                cv::Mat matImageFloat;                          // thêm ảnh training vào
                matROIResized.convertTo(matImageFloat, CV_32FC1);       // đổi Mat về float

                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       // làm phẳng

                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       // thêm vào Mat như là vector, cần thiết để phù hợp để
                                                                                            // chấp nhận kiểu dữ liệu khi gọi KNearest.train
            }   // end if
        }   // end if
    }   // end for

    std::cout << "training complete\n\n";

    // lưu phân loại vào file ///////////////////////////////////////////////////////

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);           // mở file classìication.xml

    if (fsClassifications.isOpened() == false) {                                                        // Nếu file mở thất bại
        std::cout << "error, unable to open training classifications file, exiting program\n\n";        // hiển thị thông báo lỗi
        return(0);                                                                                      // và thoát
    }

    fsClassifications << "classifications" << matClassificationInts;        // viết vào vung phân loại của file classìication
    fsClassifications.release();                                            // đóng file classìication

                                                                            // Lưu ảnh vào file image ///////////////////////////////////////////////////////

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         // mở file training image

    if (fsTrainingImages.isOpened() == false) {                                                 // Nếu file không mở đc
        std::cout << "error, unable to open training images file, exiting program\n\n";         // hiển thị thông báo lỗi
        return(0);                                                                              // thoát khỏi chương trình
    }
    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // viết vào vùng ảnh của file training images
	std::cout << matTrainingImagesAsFlattenedFloats;
	fsTrainingImages.release();                                                 // đóng file training lại

    return(0);
}




