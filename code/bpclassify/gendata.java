package map.range.classify;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * Created by bisonliao on 2017/12/3.
 */
public class Main {

    private static void genData(String filename, int cnt, boolean isARFF) throws Exception
    {
        int i;

        FileOutputStream outputStream = new FileOutputStream(filename);
        PrintStream o = new PrintStream(outputStream);
        if ( isARFF) {
            o.println("@RELATION range\n" +
                    " \n" +
                    "@ATTRIBUTE x  NUMERIC\n" +
                    "@ATTRIBUTE y   NUMERIC\n" +
                    "@ATTRIBUTE class        {1,2,3}\n" +
                    "@DATA");
        }
        for (i = 0; i < cnt; ++i) {
            double r = Math.random();
            int x = new Double(r * 65535).intValue();
            r = Math.random();
            int y = new Double(r * 65535).intValue();

            if (x < 32768 && y < 32768)
            {
                o.printf("%d,%d,1,\n", x, y);
                continue;
            }
            if (y > x)
            {
                o.printf("%d,%d,2,\n", x, y);
                continue;
            }

            o.printf("%d,%d,3,\n", x, y);


        }
        o.close();

    }

    public static void main(String args[]) throws Exception
    {
        genData("c:/range_train.csv", 100000, false);
        genData("c:/range_test.csv", 10000, false);
    }
}