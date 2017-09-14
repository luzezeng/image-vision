package com.luzz.opencv.idcard.image.enums;

public enum  ImageType {
    JPG(1, "jpg"),
    PNG(2, "png");

    private int code;
    private String type;

    ImageType(int code, String type) {
        this.code = code;
        this.type = type;
    }

    public int getCode() {
        return code;
    }

    public String getType() {
        return type;
    }
}
