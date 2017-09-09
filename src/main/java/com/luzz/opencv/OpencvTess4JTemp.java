package com.luzz.opencv;

import net.sourceforge.tess4j.ITesseract;
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
 *
 * 缺陷：1. 图片大小受限. 改进：Resize.
 *      2. 灰度化调用openCV库. 改进：后面手动采用R通道获取.
 *      3. 二值化阈值固定. 改进：后面需要对图像计算后动态获取.
 *      4. 对字符的识别调用google开源库. 改进：后面需要通过opencv-ml库自定义机器学习算法训练生成策略.
 *
 * 后面计划：学习书《学习OpenCV(中文版)》
 *
 * @author Joseph Lu
 */
public class OpencvTess4JTemp {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/IDRecognize/tempImages/";
    private static final String TESS_DATA_PATH = "/Users/joseph/workspace/IDRecognize/tessdata";
    private static final String NUMBER_PATH_IMG_NAME = "numberPart.jpg";

    public static String recognize(String imgPath) throws Exception {
        //灰度化
        Mat grayMat = new Mat();
        Mat mat = Imgcodecs.imread(imgPath);
        grayMat = getRplane(mat);
        generateImageFile(grayMat, "gray.jpg");

        //二值化
        Mat binaryMat = ostuBeresenThreshold(grayMat);
        generateImageFile(binaryMat, "binary.jpg");

        //获得标志区域
        List<RotatedRect>  rects = new ArrayList<RotatedRect>();
        Map<String, RotatedRect> rotatedRectMap = markedDetect(posDetect(grayMat, binaryMat ,rects));
        Mat outputMat = new Mat();

        //获得身份证号码字符矩阵
        normalPosArea(binaryMat, rotatedRectMap.get("ID"), outputMat);
        generateImageFile(outputMat, "numberPart.jpg");

        //获得地址字符矩阵
        normalPosArea(binaryMat, rotatedRectMap.get("Addr"), outputMat);
        generateImageFile(outputMat, "addressPart.jpg");

        //识别号码区域
        File imageFile = new File(TEMP_IMAGE_STORAGE_PATH+ NUMBER_PATH_IMG_NAME);
        ITesseract instance = new Tesseract();
        instance.setDatapath(TESS_DATA_PATH);
        instance.setLanguage("eng");
        return instance.doOCR(imageFile);
    }

    private static Mat getRplane(Mat in) {
        List<Mat> splitBGR = new ArrayList<Mat>(3); //容器大小为通道数3
        split(in,splitBGR);

        if(in.cols() > 700 || in.cols() > 600) {
            Mat resizeR = new Mat(450,600 , CV_8UC1);
            resize(splitBGR.get(2) ,resizeR ,resizeR.size());
            return resizeR;
        } else return splitBGR.get(2);
    }

    static Mat ostuBeresenThreshold(Mat in) {
        Mat out = new Mat(in.rows(), in.cols(), in.type());
        double ostu_T = threshold(in , out, 0,255 ,THRESH_OTSU); //otsu获得全局阈值
        Core.MinMaxLocResult minMaxLocResult = minMaxLoc(in);
        double min = minMaxLocResult.minVal;
        double max = minMaxLocResult.maxVal;

        double CI = 0.12;
        double beta = CI*(max - min +1)/128;
        double beta_lowT = (1-beta)*ostu_T;
        double beta_highT = (1+beta)*ostu_T;

        Mat doubleMatIn = new Mat();
        in.copyTo(doubleMatIn);
        int rows = doubleMatIn.rows();
        int cols = doubleMatIn.cols();
        double Tbn;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j ) {
                double[] p = doubleMatIn.get(i, j);
                if(i < 2 || i > rows - 3 || j < 2 || j > rows - 3) {
                    if(p[0] <= beta_lowT)
                        out.put(i, j, new double[]{0});
                    else
                        out.put(i, j, new double[]{255});
                } else {
                    Tbn = sumElems(doubleMatIn.submat(new Rect(i-2,j-2,5,5))).val[0]/25;
                    if((p[0] < beta_lowT || p[0] < Tbn) &&  (beta_lowT <= p[0] && p[0] >= beta_highT))
                        out.put(i, j, new double[]{0});
                    if( p[0] > beta_highT | (p[0] >= Tbn &&  (beta_lowT <= p[0] && p[0] >= beta_highT)))
                        out.put(i, j, new double[]{255});
                }
            }
        }
        return out;
    }

    private static List<MatOfPoint> posDetect(Mat in, Mat threshold_R, List<RotatedRect> rects) {
        Mat imgInv = new Mat(in.size(), in.type(), new Scalar(255));
        Mat threshold_Inv = new Mat();
        subtract(imgInv, threshold_R, threshold_Inv); //黑白色反转，即背景为黑色

        Mat element = getStructuringElement(MORPH_RECT ,new Size(15 ,3));  //闭形态学的结构元素
        morphologyEx(threshold_Inv, threshold_Inv, MORPH_CLOSE, element);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        //只检测外轮廓
        findContours(threshold_Inv, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        return contours;
    }

    private static Map<String, RotatedRect> markedDetect(List<MatOfPoint> contours) {
        //对候选的轮廓进行进一步筛选标记
        Map<String, RotatedRect> result = new HashMap<String, RotatedRect>();
        Iterator<MatOfPoint> iterator = contours.iterator();
        while (iterator.hasNext()) {
            MatOfPoint point = iterator.next();
            RotatedRect mr = minAreaRect(new MatOfPoint2f(point.toArray()));
            //身份证号码筛选
            if(isNumberArea(mr)) {
                result.put("ID", mr);
            }
            if (isAddrArea(mr)) {
                result.put("Addr", mr);
            }
        }
        return result;
    }

    private static boolean isNumberArea(RotatedRect candidate) {
        double error = 0.1f;
        double aspect = 4.5f/0.3f;
        int min = Double.valueOf(10 * aspect * 10).intValue();
        int max = Double.valueOf(50 * aspect * 50).intValue();
        double rmin = aspect - aspect*error;
        double rmax = aspect + aspect*error;

        int area = Double.valueOf(candidate.size.height * candidate.size.width).intValue();
        float r = (float)candidate.size.width/(float)candidate.size.height;
        if(r < 1) {
            r = 1/r;
        }
        if((area < min || area > max) || (r< rmin || r > rmax))
            return false;
        else
            return true;
    }

    private static boolean isAddrArea(RotatedRect candidate) {
        double error = 0.2f;
        double aspect = 4.6f/0.3f;
        int min = Double.valueOf(10 * aspect * 10).intValue();
        int max = Double.valueOf(50 * aspect * 50).intValue();
        double rmin = aspect - aspect*error;
        double rmax = aspect + aspect*error;

        int area = Double.valueOf(candidate.size.height * candidate.size.width).intValue();
        float r = (float)candidate.size.width/(float)candidate.size.height;
        if(r < 1)
            r = 1/r;
        if((area < min || area > max) || (r< rmin || r > rmax))
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
        Mat rotmat = getRotationMatrix2D(rects_optimal.center , angle,1);
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