package com.luzz.opencv;

import net.sourceforge.tess4j.Tesseract;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.*;

/**
 * First Version
 * 缺陷：1. 图片大小受限. 改进：Resize.
 *      2. 灰度化调用openCV库. 改进：后面手动采用R通道获取.
 *      3. 二值化阈值固定. 改进：后面需要对图像计算后动态获取.
 *      4. 对字符的识别调用google开源库. 改进：后面需要通过opencv-ml库自定义机器学习算法训练生成策略.
 *
 * 后面计划：学习书《学习OpenCV(中文版)》
 */
public class OpencvTess4J {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/IDRecognize/tempImages/";
    private static final String TESS_DATA_PATH = "/Users/joseph/workspace/IDRecognize/tessdata";
    private static final String NUMBER_PATH_IMG_NAME = "numberPart.jpg";

    public static String recognize(String imgPath) throws Exception {
        //灰度化
        Mat grayMat = new Mat();
        Mat mat = Imgcodecs.imread(imgPath);
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGB2GRAY);

        //二值化
        Mat binaryMat = new Mat();
        Imgproc.threshold(grayMat, binaryMat, 100, 255, Imgproc.THRESH_BINARY);
        generateImageFile(binaryMat, "02_binary.jpg");

        //获得身份证号码区域
        List<RotatedRect>  rects = new ArrayList<RotatedRect>();
        posDetect(grayMat, binaryMat ,rects);
        Mat outputMat = new Mat();
        normalPosArea(binaryMat, rects.get(0), outputMat); //获得身份证号码字符矩阵
        generateImageFile(outputMat, NUMBER_PATH_IMG_NAME);

        //识别号码区域
        File imageFile = new File(TEMP_IMAGE_STORAGE_PATH+ NUMBER_PATH_IMG_NAME);
        Tesseract instance = new Tesseract();
        instance.setDatapath(TESS_DATA_PATH);
        instance.setLanguage("eng");
        return instance.doOCR(imageFile);
    }

    private static void posDetect(Mat in, Mat threshold_R, List<RotatedRect> rects) {
        Mat imgInv = new Mat(in.size(), in.type(), new Scalar(255));
        Mat threshold_Inv = new Mat();
        subtract(imgInv, threshold_R, threshold_Inv); //黑白色反转，即背景为黑色

        Mat element = getStructuringElement(MORPH_RECT ,new Size(15 ,3));  //闭形态学的结构元素
        morphologyEx(threshold_Inv, threshold_Inv, MORPH_CLOSE, element);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        findContours(threshold_Inv, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);//只检测外轮廓

        //对候选的轮廓进行进一步筛选
        Iterator<MatOfPoint> iterator = contours.iterator();
        while (iterator.hasNext()) {
            MatOfPoint point = iterator.next();
            RotatedRect mr = minAreaRect(new MatOfPoint2f(point.toArray()));
            if(isEligible(mr)) {
                rects.add(mr);
            }
        }
    }

    private static boolean isEligible(RotatedRect candidate) {
        double error = 0.2f;
        double aspect = 4.5f/0.3f; //长宽比
        int min = Double.valueOf(10 * aspect * 10).intValue(); //最小区域
        int max = Double.valueOf(50 * aspect * 50).intValue();  //最大区域
        double rmin = aspect - aspect*error; //考虑误差后的最小长宽比
        double rmax = aspect + aspect*error; //考虑误差后的最大长宽比

        int area = Double.valueOf(candidate.size.height * candidate.size.width).intValue();
        float r = (float)candidate.size.width/(float)candidate.size.height;
        if(r <1)
            r = 1/r;

        if( (area < min || area > max) || (r< rmin || r > rmax)  ) //满足该条件才认为该candidate为车牌区域
            return false;
        else
            return true;
    }

    private static void normalPosArea(Mat intputImg, RotatedRect rects_optimal, Mat output_area) {
        double r,angle;

        angle = rects_optimal.angle;
        r = rects_optimal.size.width / rects_optimal.size.height;
        if(r<1)
            angle = 90 + angle;
        Mat rotmat = getRotationMatrix2D(rects_optimal.center , angle,1);//获得变形矩阵对象
        Mat img_rotated = new Mat();
        warpAffine(intputImg ,img_rotated,rotmat, intputImg.size(),INTER_CUBIC);

        //裁剪图像
        Size rect_size = rects_optimal.size;

        if(r<1) {
            double tmp = rect_size.width;
            rect_size.width = rect_size.height;
            rect_size.height = tmp;
        }
        Mat  img_crop = new Mat();
        getRectSubPix(img_rotated ,rect_size,rects_optimal.center , img_crop );

        //用光照直方图调整所有裁剪得到的图像，使具有相同宽度和高度，适用于训练和分类
        Mat resultResized = new Mat();
        resultResized.create(20,300,CV_8UC1);
        resize(img_crop , resultResized,resultResized.size() , 0,0,INTER_CUBIC);
        resultResized.copyTo(output_area);

    }

    private static void generateImageFile(Mat mat, String imageName) throws IOException {
        byte[] data = new byte[mat.rows() * mat.cols() * (int)(mat.elemSize())];
        mat.get(0, 0, data);
        BufferedImage img = new BufferedImage(mat.cols(), mat.rows(),BufferedImage.TYPE_BYTE_GRAY);
        img.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
        ImageIO.write(img, "jpg", new File(TEMP_IMAGE_STORAGE_PATH + imageName));
    }
}
