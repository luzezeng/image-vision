package com.luzz.opencv.idcard.handler.ml.enums;

public enum LanguageType {
    ENGLIST(1, "eng");

    private int code;
    private String type;

    LanguageType(int code, String type) {
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
