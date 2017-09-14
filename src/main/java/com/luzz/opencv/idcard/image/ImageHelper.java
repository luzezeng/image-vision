package com.luzz.opencv.idcard.image;

import com.luzz.opencv.idcard.image.enums.IDCard;
import com.luzz.opencv.idcard.image.enums.ImageType;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.imgproc.Imgproc.minAreaRect;

/**
 * Tools to handle image
 *
 * @author Joseph
 */
public class ImageHelper {
    /**
     * Normal way to generate gray scale for image
     * @param image
     * @return
     * @throws Exception
     */
    public static Mat generateGray(Mat image) throws Exception {
        if (image == null || image.channels() != 3) {
            throw new Exception("Image is null or channel count isn't three");
        }
        Mat grayMat = new Mat();
        Imgproc.cvtColor(image, grayMat, Imgproc.COLOR_RGB2GRAY);
        return grayMat;
    }

    /**
     * Obtain R-channel of image
     * @param image
     * @return
     */
    public static Mat obtainRChannel(Mat image) {
        List<Mat> splitBGR = new ArrayList<Mat>(3);
        split(image, splitBGR);
        if(image.cols() > 700 || image.cols() > 600) {
            Mat resizeR = new Mat(450,600 , CV_8UC1);
            resize(splitBGR.get(2) ,resizeR ,resizeR.size());
            return resizeR;
        } else {
            return splitBGR.get(2);
        }
    }

    /**
     * Obtain G-channel of image
     * @param image
     * @return
     */
    public static Mat obtainGChannel(Mat image) {
        List<Mat> splitBGR = new ArrayList<Mat>(3);
        split(image, splitBGR);
        if(image.cols() > 700 || image.cols() > 600) {
            Mat resizeR = new Mat(450,600 , CV_8UC1);
            resize(splitBGR.get(1) ,resizeR ,resizeR.size());
            return resizeR;
        } else {
            return splitBGR.get(1);
        }
    }

    /**
     * Obtain B-channel of image
     * @param image
     * @return
     */
    public static Mat obtainBChannel(Mat image) {
        List<Mat> splitBGR = new ArrayList<Mat>(3);
        split(image, splitBGR);
        if(image.cols() > 700 || image.cols() > 600) {
            Mat resizeR = new Mat(450,600 , CV_8UC1);
            resize(splitBGR.get(0) ,resizeR ,resizeR.size());
            return resizeR;
        } else {
            return splitBGR.get(0);
        }
    }

    /**
     * Normal way to generate binary for gray scale
     * @param gray
     * @return
     * @throws Exception
     */
    public static Mat generateNormalBinary(Mat gray) throws Exception {
        if (gray == null || gray.channels() != 1) {
            throw new Exception("Image is null or channel count is not one");
        }
        Mat binaryMat = new Mat();
        Imgproc.threshold(gray, binaryMat, 100, 255, Imgproc.THRESH_BINARY);
        return binaryMat;
    }

    /**
     * A type of self-adaption way to generate binary
     * @param gray
     * @return
     */
    public static Mat generateSelfAdaptionBinary(Mat gray) {
        Mat out = new Mat(gray.rows(), gray.cols(), gray.type());
        double ostu_T = threshold(gray , out, 0,255 ,THRESH_OTSU);
        Core.MinMaxLocResult minMaxLocResult = minMaxLoc(gray);
        double min = minMaxLocResult.minVal;
        double max = minMaxLocResult.maxVal;

        double CI = 0.12;
        double beta = CI*(max - min +1)/128;
        double beta_lowT = (1-beta)*ostu_T;
        double beta_highT = (1+beta)*ostu_T;

        Mat doubleMatIn = new Mat();
        gray.copyTo(doubleMatIn);
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

    /**
     * find attributes of IDCard
     * @param contours
     * @return
     */
    public static Map<IDCard, RotatedRect> findIDCardAttribute(List<MatOfPoint> contours) {
        Map<IDCard, RotatedRect> result = new HashMap<IDCard, RotatedRect>();
        Iterator<MatOfPoint> iterator = contours.iterator();
        while (iterator.hasNext()) {
            MatOfPoint point = iterator.next();
            RotatedRect mr = minAreaRect(new MatOfPoint2f(point.toArray()));
            if(isNumberArea(mr)) {
                result.put(IDCard.ID, mr);
            }
            if (isAddressArea(mr)) {
                result.put(IDCard.ADDRESS, mr);
            }
        }
        return result;
    }

    /**
     * Detect contours
     * @param gray
     * @param binary
     * @return
     */
    public static List<MatOfPoint> detectContours(Mat gray, Mat binary) {
        Mat imgInv = new Mat(gray.size(), gray.type(), new Scalar(255));
        Mat thresholdInv = new Mat();
        subtract(imgInv, binary, thresholdInv);

        Mat element = getStructuringElement(MORPH_RECT ,new Size(15 ,3));
        morphologyEx(thresholdInv, thresholdInv, MORPH_CLOSE, element);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        findContours(thresholdInv, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        return contours;
    }

    /**
     * Cut the area of rectsOptimal
     * @param intputImg
     * @param rectsOptimal
     */
    public static Mat cutArea(Mat intputImg, RotatedRect rectsOptimal) {
        Mat outputArea = new Mat();
        double r,angle;
        angle = rectsOptimal.angle;
        r = rectsOptimal.size.width / rectsOptimal.size.height;
        if(r < 1) {
            angle = 90 + angle;
        }
        Mat rotMat = getRotationMatrix2D(rectsOptimal.center , angle,1);
        Mat imgRotated = new Mat();
        warpAffine(intputImg ,imgRotated,rotMat, intputImg.size(),INTER_CUBIC);

        //cut the image
        Size rectSize = rectsOptimal.size;

        if(r<1) {
            double tmp = rectSize.width;
            rectSize.width = rectSize.height;
            rectSize.height = tmp;
        }
        Mat  imgCrop = new Mat();
        getRectSubPix(imgRotated ,rectSize,rectsOptimal.center , imgCrop );

        Mat resultResized = new Mat();
        resultResized.create(20,300,CV_8UC1);
        resize(imgCrop , resultResized,resultResized.size() , 0,0,INTER_CUBIC);
        resultResized.copyTo(outputArea);
        return outputArea;
    }

    /**
     * Generate gray image file
     * @param mat
     * @param imagePath
     * @param type
     * @throws IOException
     */
    public static void generateGrayImageFile(Mat mat, String imagePath, ImageType type) throws IOException {
        byte[] data = new byte[mat.rows() * mat.cols() * (int)(mat.elemSize())];
        mat.get(0, 0, data);
        BufferedImage img = new BufferedImage(mat.cols(), mat.rows(),BufferedImage.TYPE_BYTE_GRAY);
        img.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
        ImageIO.write(img, type.getType(), new File(imagePath));
    }

    /**
     * cut every char of area chosen
     * @param inputImg
     */
    public static List<Mat> cutChar(Mat inputImg) {
        List<Mat> result = new ArrayList<Mat>();
        Mat img_threshold = new Mat();
        Mat whiteImg = new Mat(inputImg.size(),inputImg.type(),new Scalar(255));
        Mat inInv = new Mat();
        subtract(whiteImg, inputImg, inInv);
        threshold(inInv ,img_threshold , 0,255 ,THRESH_OTSU );
        int xChar[] = new int[19];
        short counter = 1;
        short num = 0;
        boolean flag[] = new boolean[img_threshold.cols()];
        for (int j = 0; j < img_threshold.cols(); ++j) {
            flag[j] = true;
            for(int i = 0; i < img_threshold.rows(); ++i) {
                if(img_threshold.get(i,j)[0] != 0 ) {
                    flag[j] = false;
                    break;
                }

            }
        }
        for (int i = 0; i < img_threshold.cols() - 2; ++i) {
            if(flag[i] == true) {
                xChar[counter] += i;
                num++;
                if(flag[i+1] ==false && flag[i+2] ==false) {
                    xChar[counter] = xChar[counter] / num;
                    num = 0;
                    counter++;
                }
            }
        }
        xChar[18] = img_threshold.cols();
        for (int i = 0; i < 18; i++) {
            result.add(new Mat(inInv, new Rect(xChar[i],0, xChar[i+1] - xChar[i], img_threshold.rows())));
        }
        return result;
    }

    private static boolean isNumberArea(RotatedRect candidate) {
        double error = 0.1f;
        double aspect = 4.5f/0.3f;
        int min = Double.valueOf(10 * aspect * 10).intValue();
        int max = Double.valueOf(50 * aspect * 50).intValue();
        double rMin = aspect - aspect*error;
        double rMax = aspect + aspect*error;

        int area = Double.valueOf(candidate.size.height * candidate.size.width).intValue();
        float r = (float)candidate.size.width/(float)candidate.size.height;
        if(r < 1) {
            r = 1/r;
        }
        if((area < min || area > max) || (r< rMin || r > rMax))
            return false;
        else
            return true;
    }

    private static boolean isAddressArea(RotatedRect candidate) {
        double error = 0.2f;
        double aspect = 4.6f/0.3f;
        int min = Double.valueOf(10 * aspect * 10).intValue();
        int max = Double.valueOf(50 * aspect * 50).intValue();
        double rmin = aspect - aspect*error;
        double rmax = aspect + aspect*error;

        int area = Double.valueOf(candidate.size.height * candidate.size.width).intValue();
        float r = (float)candidate.size.width/(float)candidate.size.height;
        if(r < 1)
            r = 1 / r;
        if((area < min || area > max) || (r< rmin || r > rmax))
            return false;
        else
            return true;
    }
}
