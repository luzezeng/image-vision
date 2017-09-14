package com.luzz.opencv.idcard;

import java.util.Map;

/**
 * Recognize numbers and words in identity card of Chinese
 *
 * @author Joseph Lu
 */
public interface IDCardClient {

    /**
     * return the information
     * @param imgPath
     * @return
     */
    Map<String, String> recognize(String imgPath) throws Exception;

}
