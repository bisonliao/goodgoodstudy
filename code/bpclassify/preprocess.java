package bank.marketing;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.concurrent.ExecutionException;


public class preprocess {

   //�����ֶ���Ϣ����
    public static class FieldDesc
    {
        public enum FieldType { //�ֶ�����ȡֵ��Χ
            UINT63,
            INT64,
            DOUBLE,
            ENUM_STR,//ȡֵ��ö�ٵ��ַ���
        };
        public FieldType fieldType;
        //���ڽ�UINT63�������ֵ���͵��ֶι�һ����UINT63ȡֵ��Χ�����ֵ��Сֵ
        public long minValue = 0;
        public long maxValue = Long.MAX_VALUE;

        //����ö�����ַ���ת��Ϊ��ֵ
        private HashMap<String,  Long> strValList = new HashMap<String, Long>();

        public long GetUnitedValue(String s)
        {
            if (fieldType == FieldType.ENUM_STR) {
                Long v = strValList.get(s);
                if (v == null) {
                    System.err.println("WARNING:mismatch entry " + s);
                    return Long.MAX_VALUE;
                }
                return v.longValue();
            }
            if (fieldType == FieldType.DOUBLE || fieldType == FieldType.INT64)
            {
                double input = new Double(s).doubleValue();

                double v = (input - minValue) / (maxValue-minValue) * Long.MAX_VALUE;
                return new Double(v).longValue();
            }
            if (fieldType == FieldType.UINT63)
            {
                return new Long(s).longValue();
            }
            System.err.println("ERROR:invalid field type:"+fieldType);
            return Long.MAX_VALUE;
        }


        public FieldDesc(FieldType type, Long min, Long max, String[] strList)
        {
            fieldType = type;

            if (min != null) {  minValue = min.longValue();}
            if (max != null) {  maxValue = max.longValue();}


            if (type == FieldType.ENUM_STR)
            {
                if (  strList != null && strList.length > 0) {
                    //�����п���ȡֵ��һ���� UINT63�Ŀռ�
                    int i;
                    for (i = 0; i < strList.length; ++i) {
                        double index = i;
                        index = index / strList.length * Long.MAX_VALUE;
                        strValList.put(strList[i], new Long(new Double(index).longValue()));
                    }
                }
                else
                {
                    System.err.println("ERROR:faild to initiate field desc ");
                }
            }

        }

    };

    //����ͬ�����ݼ��ϣ���Ҫ��д����������飬����ÿ���ֶε�����
    // �� UCI���ݼ�AdultΪ���� https://archive.ics.uci.edu/ml/datasets/adult
            public static FieldDesc[] fieldDescs = {
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR, null, null, new String[]{"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"}),
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR,null, null, new String[]{"Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"}),
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR, null, null, new String[]{"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse", "?"}),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR, null, null, new String[]{"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"}),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR,  null, null, new String[]{"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"}),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR, null, null, new String[]{"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"}),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR,  null, null, new String[]{"Female", "Male"}),
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
                    new FieldDesc(FieldDesc.FieldType.ENUM_STR, null, null, new String[]{"United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"}),
                    new FieldDesc(FieldDesc.FieldType.UINT63, null, null, null),
            };






    public static void convert(String srcFile, String destFile, String sepRegex) throws Exception
    {

        int i;


        BufferedReader r = new BufferedReader(new FileReader(srcFile));
        BufferedWriter w = new BufferedWriter(new FileWriter(destFile));

        int fieldsNumOfFirstLine = -1;

        StringBuffer sb = new StringBuffer();

        while (true) {
            if (sb.length() > 0) {
                sb.delete(0, sb.length());
            }
            String s = r.readLine();
            if (s == null) {
                break;
            }
            String[] fields = s.split(sepRegex);
            if (fields == null)
            {
                System.err.println("ERROR:fail to split into fields:"+s);
                continue;
            }
            if (fieldsNumOfFirstLine < 0)
            {
                fieldsNumOfFirstLine = fields.length;
            }
            if (fields.length != fieldsNumOfFirstLine)
            {
                System.err.println("ERROR: field number mismatch!"+s);
                continue;
            }

            for (i = 0; i < fields.length; ++i) {

                    long value = fieldDescs[i].GetUnitedValue(fields[i]);
                    sb.append("" + value + ",");
            }
            sb.append("\n");
            w.write(sb.toString());



        }
        r.close();
        w.close();
    }

    public static void main(String[] args) {
	// write your code here
        try {
            convert("d:\\MLDATA\\adult.data", "d:\\MLDATA\\adult_standard.csv", "[ \t]*,[ \t]*");
            convert("d:\\MLDATA\\adult.test", "d:\\MLDATA\\adult_standard_test.csv", "[ \t]*,[ \t]*");
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
