package com.luzz.opencv.idcard.ml;

import com.luzz.opencv.idcard.ml.enums.LanguageType;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.TrainData;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.countNonZero;
import static org.opencv.core.Core.minMaxLoc;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.Mat.zeros;
import static org.opencv.imgproc.Imgproc.filter2D;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.ml.ANN_MLP.UPDATE_WEIGHTS;
import static org.opencv.ml.Ml.ROW_SAMPLE;

public class Recognizor {
    private static String TESS_DATA_PATH = "/Users/joseph/workspace/image-vision/src/main/resources/tessdata";
    private static final String TRAIN_NUMBER_CHAR_SET_PATH = "/Users/joseph/workspace/image-vision/src/main/resources/trainnums/";
    private static final String ANN_XML_PATH = "/Users/joseph/workspace/image-vision/src/main/resources/annconfig.xml";

    /**
     * Recognize by Tess4Jer
     * @param image
     * @return
     */
    public static String recognizeByTess4J(File image, LanguageType language) throws Exception {
        ITesseract instance = new Tesseract();
        instance.setDatapath(TESS_DATA_PATH);
        instance.setLanguage(language.getType());
        return instance.doOCR(image);
    }

    /**
     * recognizeIDCardByCNN
     * @param source
     * @return
     */
    public static List<String> recognizeIDCardByCNN(List<Mat> source) {
        generateAnnConfig();
        ANN_MLP ann = ANN_MLP.create();
        annTrainForIDNumber(ann, 10, 24);
        return classifyForIDNumber(ann, source);
    }


    private static void generateAnnConfig() {
        Mat trainData = new Mat();
        //1700*48维，也即每个样本有48个特征值
        Mat classes = zeros(1,550, CV_8UC1);
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
        List<Double> feat = new ArrayList<Double>();
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

    private static void annTrainForIDNumber(ANN_MLP ann, int numCharacters, int nlayers) {
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

    private static List<String> classifyForIDNumber(ANN_MLP ann, List<Mat> charMat) {
        List<String> char_result = new ArrayList<String>(charMat.size());
        for (int i=0; i < charMat.size(); i++) {
            Mat output = new Mat(1 ,10 , CV_32FC1);
            Mat char_feature = new Mat();
            calcGradientFeat(charMat.get(i) ,char_feature);
            ann.predict(char_feature ,output, UPDATE_WEIGHTS);
            Core.MinMaxLocResult minMaxLocResult = minMaxLoc(output);
            Point maxLoc = minMaxLocResult.maxLoc;
            char_result.set(i, String.valueOf(maxLoc.x));
        }
        return char_result;
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
}
