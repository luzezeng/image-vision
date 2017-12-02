package com.luzz.opencv.idcard.impl;

import com.luzz.opencv.idcard.IDCardClient;
import com.luzz.opencv.idcard.handler.image.ImageHandler;
import com.luzz.opencv.idcard.handler.image.enums.IDCard;
import com.luzz.opencv.idcard.handler.image.enums.ImageType;
import com.luzz.opencv.idcard.handler.ml.MLHandler;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

/**
 * Second Version
 *
 * Some tips:
 *      1. Have been improved in every item mentioned in the last version
 *      2. Still the annconfig.xml file's generating and reading has not finished
 *      3. Will add other character's recognization
 *      4. Will improve the accuracy
 *      5. Will try other machine learning method
 * Later plan：Learn machine learning and opencv
 *
 * @author Joseph Lu
 */
public class CNNer implements IDCardClient {
    private static final String TEMP_IMAGE_STORAGE_PATH = "E:/workspace/image-vision/tempimages";

    public Map<String, String> recognize(String imgPath) throws Exception {
        //gray from R channel
        Mat grayMat = ImageHandler.obtainRChannel(Imgcodecs.imread(imgPath));
        ImageHandler.generateGrayImageFile(grayMat, TEMP_IMAGE_STORAGE_PATH + "gray.jpg", ImageType.JPG);

        //binary from self-adaption
        Mat binaryMat = ImageHandler.generateSelfAdaptionBinary(grayMat);
        ImageHandler.generateGrayImageFile(binaryMat, TEMP_IMAGE_STORAGE_PATH + "binary.jpg", ImageType.JPG);

        //get ID card attributes
        Map<IDCard, RotatedRect> rotatedRectMap = ImageHandler.findIDCardAttribute(ImageHandler.detectContours(grayMat, binaryMat));

        //get number part
        Mat IdArea = ImageHandler.cutArea(binaryMat, rotatedRectMap.get(IDCard.ID));
        ImageHandler.generateGrayImageFile(IdArea, TEMP_IMAGE_STORAGE_PATH + "numberPart.jpg", ImageType.JPG);

        //Cut the number mat
        List<Mat> charMat = ImageHandler.cutChar(IdArea);
        for (int i = 0; i < charMat.size(); i ++) {
            ImageHandler.generateGrayImageFile(charMat.get(i), TEMP_IMAGE_STORAGE_PATH + "/chars/" + i + ".jpg", ImageType.JPG);
        }

        final StringBuffer idBuffer = new StringBuffer();
        for (String item : MLHandler.recognizeIDCardByCNN(charMat)) {
            if (item != null && !item.equals("")) {
                idBuffer.append(item);
            }
        }

        return new HashMap<String, String>() {
            {
                put("ID", idBuffer.toString());
            }
        };
    }
}
