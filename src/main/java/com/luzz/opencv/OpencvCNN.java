package com.luzz.opencv;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.TrainData;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.Mat.zeros;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.ml.ANN_MLP.UPDATE_WEIGHTS;
import static org.opencv.ml.Ml.ROW_SAMPLE;

/**
 * Second Version
 *
 * Some tips:
 *      1. Have been improved in every item mentioned in the last version
 *      2. Still the annconfig.xml file's generating and reading has not finished
 *      3. Will add other character's recognization
 *      4. Will improve the accuracy
 *      5. Will try other machine learning method
 * 后面计划：学习书《学习OpenCV(中文版)》
 *
 * @author Joseph Lu
 */
public class OpencvCNN {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/IDRecognize/tempImages/";
    private static final String TRAIN_NUMBER_CHAR_SET_PATH = "/Users/joseph/workspace/IDRecognize/src/main/resources/trainnums/";
    private static final String ANN_XML_PATH = "/Users/joseph/workspace/IDRecognize/src/main/resources/annconfig.xml";

    public static String recognize(String imgPath) throws Exception {
        Mat mat = Imgcodecs.imread(imgPath);
        System.out.println(mat.type()+ " " + mat.channels());

        //User the channel R as the gray picture
        Mat imgRplane = getRplane(mat);
        generateImageFile(imgRplane, "r.jpg");

        //Find the number area
        List<RotatedRect>  rects = new ArrayList<RotatedRect>();
        posDetect(imgRplane ,rects);
        Mat outputMat = new Mat();
        normalPosArea(imgRplane, rects.get(0), outputMat);

        //Cut the number mat
        List<Mat> char_mat = new ArrayList<Mat>();
        char_segment(outputMat , char_mat);
        for (int i = 0; i < char_mat.size(); i ++) {
            generateImageFile(char_mat.get(i), "/chars/" + i + ".jpg");
        }

        //Generate annconfig.xml
        getAnnXML();

        //Choose ann as trained theory
        ANN_MLP ann = ANN_MLP.create();
        ann_train(ann, 10, 24);
        List<Integer> char_result = new ArrayList<Integer>();
        classify(ann ,char_mat ,char_result);
        return String.valueOf(char_result);
    }

    private static Mat getRplane(Mat in) {
        List<Mat> splitBGR = new ArrayList<Mat>(3);
        split(in,splitBGR);
        if(in.cols() > 700 || in.cols() > 600) {
            Mat resizeR = new Mat(450,600 , CV_8UC1);
            resize(splitBGR.get(2) ,resizeR ,resizeR.size());
            return resizeR;
        }
        else {
            return splitBGR.get(2);
        }
    }

    private static void posDetect(Mat in, List<RotatedRect> rects) throws Exception {
        Mat threshold_R = new Mat(in.rows(), in.cols(), in.type());
        OstuBeresenThreshold(in ,threshold_R);
        generateImageFile(threshold_R, "threshold_R.jpg");
        Mat imgInv = new Mat(in.size(), in.type(), new Scalar(255));
        Mat threshold_Inv = new Mat();

        //turn over the back color and front
        subtract(imgInv, threshold_R, threshold_Inv);
        generateImageFile(threshold_Inv, "threshold_Inv.jpg");

        //getStructuringElement
        Mat element = getStructuringElement(MORPH_RECT ,new Size(15 ,3));
        morphologyEx(threshold_Inv, threshold_Inv, MORPH_CLOSE, element);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        //findContours
        findContours(threshold_Inv, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        //iterate the contours
        Iterator<MatOfPoint> iterator = contours.iterator();
        while (iterator.hasNext()) {
            MatOfPoint point = iterator.next();
            RotatedRect mr = minAreaRect(new MatOfPoint2f(point.toArray()));
            if(isEligible(mr)) {
                rects.add(mr);
            }
        }
    }

    private static void OstuBeresenThreshold(Mat in, Mat out) {
        System.out.println("intput channels :" + in.channels());
        double ostu_T = threshold(in , out, 0,255 ,THRESH_OTSU);
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
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                double[] p = doubleMatIn.get(i, j);
                if(i <2 || i>rows - 3 || j<2 || j>rows - 3) {
                    if(p[0] <= beta_lowT) {
                        out.put(i, j, new double[]{0});
                    } else {
                        out.put(i, j, new double[]{255});
                    }
                } else {
                    Tbn = sumElems(doubleMatIn.submat(new Rect(i-2,j-2,5,5))).val[0]/25;
                    if((p[0] < beta_lowT || p[0] < Tbn) &&  (beta_lowT <= p[0] && p[0] >= beta_highT)) {
                        out.put(i, j, new double[]{0});
                    }
                    if( p[0] > beta_highT | (p[0] >= Tbn &&  (beta_lowT <= p[0] && p[0] >= beta_highT))) {
                        out.put(i, j, new double[]{255});
                    }
                }
            }
        }
    }

    private static boolean isEligible(RotatedRect candidate) {
        double error = 0.2f;
        //长宽比
        double aspect = 4.5f/0.3f;
        //最小区域
        int min = Double.valueOf(10 * aspect * 10).intValue();
        //最大区域
        int max = Double.valueOf(50 * aspect * 50).intValue();
        //考虑误差后的最小长宽比
        double rmin = aspect - aspect*error;
        //考虑误差后的最大长宽比
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

    private static void normalPosArea(Mat intputImg, RotatedRect rects_optimal, Mat output_area) {
        double r,angle;
        angle = rects_optimal.angle;
        r = rects_optimal.size.width / rects_optimal.size.height;
        if(r < 1) {
            angle = 90 + angle;
        }
        Mat rotmat = getRotationMatrix2D(rects_optimal.center , angle,1);//获得变形矩阵对象
        Mat img_rotated = new Mat();
        warpAffine(intputImg ,img_rotated,rotmat, intputImg.size(),INTER_CUBIC);

        //裁剪图像
        Size rect_size = rects_optimal.size;
        if(r < 1) {
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

    private static void char_segment(Mat inputImg, List<Mat> dst_mat) {
        Mat img_threshold = new Mat();
        Mat whiteImg = new Mat(inputImg.size(),inputImg.type(),new Scalar(255));
        Mat in_Inv = new Mat();
        subtract(whiteImg, inputImg, in_Inv);
        threshold(in_Inv ,img_threshold , 0,255 ,THRESH_OTSU );
        int x_char[] = new int[19];
        short counter = 1;
        short num = 0;
        boolean flag[] = new boolean[img_threshold.cols()];
        for(int j = 0; j < img_threshold.cols(); ++j) {
            flag[j] = true;
            for(int i = 0; i < img_threshold.rows(); ++i) {
                if(img_threshold.get(i,j)[0] != 0 ) {
                    flag[j] = false;
                    break;
                }

            }
        }
        for(int i = 0; i < img_threshold.cols() - 2; ++i) {
            if(flag[i] == true) {
                x_char[counter] += i;
                num++;
                if(flag[i+1] ==false && flag[i+2] ==false) {
                    x_char[counter] = x_char[counter]/num;
                    num = 0;
                    counter++;
                }
            }
        }
        x_char[18] = img_threshold.cols();
        for(int i = 0;i < 18;i++) {
            dst_mat.add(new Mat(in_Inv , new Rect(x_char[i],0, x_char[i+1] - x_char[i] ,img_threshold.rows())));
        }
    }

    private static void getAnnXML() {
        Mat  trainData = new Mat();
        Mat classes = zeros(1,550, CV_8UC1);   //1700*48维，也即每个样本有48个特征值
        String annXmlPath = ANN_XML_PATH;
        String numberCharPath = TRAIN_NUMBER_CHAR_SET_PATH;
        Mat img_read;
        for (int i = 0; i<10 ;i++) {
            for (int j=1; j < 51; j++) {
                //todo:Read all pictures in directory number_char one by one
                img_read = Imgcodecs.imread(numberCharPath + "/" + i + "i(" + j + ").png", 0);
                Mat dst_feature = new Mat();
                calcGradientFeat(img_read, dst_feature);
                trainData.push_back(dst_feature);
                classes.put(0, i*50 + j -1, i);
            }
        }
        //todo:Write all trainData and classes in annconfig.xml
    }

    private static void calcGradientFeat(Mat imgSrc, Mat out) {
        List<Double>  feat = new ArrayList<Double>();
        Mat image = new Mat();
        resize(imgSrc, image, new Size(8,16));

        // calculate bur follow x or y
        double [][] mask = new double[][] { { 1.0, 2.0, 1.0 }, { 0.0, 0.0, 0.0 }, { -1.0, -2.0, -1.0 } };
        Mat maskMat = new Mat(3, 3, CvType.CV_32F);
        for (int i = 0; i < maskMat.rows(); i ++) {
            maskMat.put(i, 0, mask[i]);
        }

        Mat y_mask = new Mat();
        Core.divide(maskMat, Scalar.all(8), y_mask);
        Mat x_mask = y_mask.t();
        Mat sobelX = new Mat();
        Mat sobelY = new Mat();

        filter2D(image, sobelX, CvType.CV_32F, x_mask);
        filter2D(image, sobelY, CvType.CV_32F, y_mask);

        Core.absdiff(sobelX, Scalar.all(0), sobelX);
        Core.absdiff(sobelY, Scalar.all(0), sobelY);

        double totleValueX = sumMatValue(sobelX);
        double totleValueY = sumMatValue(sobelY);

        // divide the image to subImages as formed 4*2, and calculate every subImage's sum of gray value and percentage
        for (int i = 0; i < image.rows(); i = i + 4) {
            for (int j = 0; j < image.cols(); j = j + 4) {
                Mat subImageX = sobelX.submat(new Rect(j, i, 4, 4));
                feat.add(sumMatValue(subImageX) / totleValueX);
                Mat subImageY= sobelY.submat(new Rect(j, i, 4, 4));
                feat.add(sumMatValue(subImageY) / totleValueY);
            }
        }

        // calculate the second feature
        Mat imageGray = new Mat();
        resize(imgSrc, imageGray, new Size(4,8));
        Mat p = imageGray.reshape(1,1);
        p.convertTo(p,CV_32FC1);
        for (int i = 0; i < p.cols(); i++ ) {
            feat.add(p.get(0, i)[0]);
        }

        //水平直方图
        Mat vhist = projectHistogram(imgSrc , 1);
        //垂直直方图
        Mat hhist = projectHistogram(imgSrc , 0);
        for (int i = 0; i < vhist.cols(); i++) {
            feat.add(vhist.get(0, i)[0]);
        }
        for (int i = 0; i < hhist.cols(); i++) {
            feat.add(hhist.get(0, i)[0]);
        }

        out = zeros(1, feat.size() , CvType.CV_32F);
        for (int i = 0; i < feat.size(); i++) {
            out.put(0, i, feat.get(i));
        }
    }

    private static double sumMatValue(Mat image) {
        float sumValue = 0;
        int r = image.rows();
        int c = image.cols();
        if (image.isContinuous()) {
            c = r*c;
            r = 1;
        }
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                double[] p = image.get(i, j);
                sumValue += p[0];
            }
        }
        return sumValue;
    }

    private static Mat projectHistogram(Mat img, int t) {
        Mat lowData = new Mat();
        resize(img, lowData, new Size(8 ,16 )); //缩放到8*16

        int sz = (t != 0) ? lowData.rows() : lowData.cols();
        Mat mhist = zeros(1, sz, CvType.CV_32F);

        for(int j = 0; j < sz; j++) {
            Mat data = (t != 0)? lowData.row(j) : lowData.col(j);
            mhist.put(0, j, countNonZero(data));
        }

        Core.MinMaxLocResult minMaxLocResult = minMaxLoc(mhist);
        double max = minMaxLocResult.maxVal;
        if(max > 0) {
            mhist.convertTo(mhist ,-1,1.0f/max , 0);
        }
        return mhist;
    }

    private static void ann_train(ANN_MLP ann, int numCharacters, int nlayers) {
        Mat trainData = new Mat();
        Mat classes = new Mat();
        //todo:read the annconfig.xml and put all the data into the two data set below
        //3层神经网络
        Mat layerSizes = new Mat(1, 3 , CV_32SC1);
        //输入层的神经元结点数，设置为24
        layerSizes.put(0, 0, trainData.cols());
        //1个隐藏层的神经元结点数，设置为24
        layerSizes.put(0, 1, nlayers);
        //输出层的神经元结点数为:10
        layerSizes.put(0, 2, numCharacters);
        ann.setLayerSizes(layerSizes);

        Mat trainClasses = new Mat();
        trainClasses.create(trainData.rows() , numCharacters ,CV_32FC1);
        for (int i = 0; i < trainData.rows(); i++) {
            for (int k=0 ; k < trainClasses.cols() ; k++ ) {
                if ( k == (int)classes.get(0, i)[0]) {
                    trainClasses.put(i,k, 1);
                } else {
                    trainClasses.put(i,k, 0);
                }
            }

        }
        TrainData data = TrainData.create(trainData, ROW_SAMPLE, classes);
        ann.train(data);
    }

    private static void classify(ANN_MLP ann, List<Mat> char_Mat, List<Integer> char_result) {
        char_result = new ArrayList<Integer>(char_Mat.size());
        for (int i=0;i<char_Mat.size(); ++i) {
            Mat output = new Mat(1 ,10 , CV_32FC1); //1*10矩阵
            Mat char_feature = new Mat();
            calcGradientFeat(char_Mat.get(i) ,char_feature);
            ann.predict(char_feature ,output, UPDATE_WEIGHTS);
            Core.MinMaxLocResult minMaxLocResult = minMaxLoc(output);
            Point maxLoc = minMaxLocResult.maxLoc;
            char_result.set(i, (int)maxLoc.x);
        }
    }

    /**
     * To generate the last char specially for it didn't perform very well in that case
     * @param char_result
     */
    private static void getParityBit(List<Integer> char_result) {
        int mod = 0;
        int[] wights = { 7,9,10,5,8,4 ,2,1,6,3,7,9,10,5,8,4,2};
        for(int i =0; i < 17 ;++i)
            mod += char_result.get(i) * wights[i];//乘相应系数求和
        mod = mod%11; //对11求余
        int [] value = new int[]{1,0,10,9,8,7,6,5,4,3,2};
        char_result.set(17, value[mod]);
    }

    private static void generateImageFile(Mat mat, String imageName) throws IOException {
        byte[] data = new byte[mat.rows() * mat.cols() * (int)(mat.elemSize())];
        mat.get(0, 0, data);
        BufferedImage img = new BufferedImage(mat.cols(), mat.rows(),BufferedImage.TYPE_BYTE_GRAY);
        img.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
        ImageIO.write(img, "jpg", new File(TEMP_IMAGE_STORAGE_PATH + imageName));
    }
}
