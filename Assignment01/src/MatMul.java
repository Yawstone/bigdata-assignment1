import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;

/**
 * Basic dense matrix multiplication, triple loop (i-j-k), C = A * B.
 * Storage: double[n][n] (array of row arrays).
 * Usage: java MatMul <input.bin> <n> <reps> <warmup> [output.bin]
 * Timer covers only the triple loop. File loading and allocation are outside.
 */
public class MatMul {
    static void multiply(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            double[] Ai = A[i];
            double[] Ci = C[i];
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += Ai[k] * B[k][j];
                }
                Ci[j] = sum;
            }
        }
    }

    static double[][] readMatrix(ByteBuffer buf, int n) {
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i][j] = buf.getDouble();
        return M;
    }

    /** Peak resident set size of this process (Linux VmHWM, kB); -1 if unavailable. */
    static long peakRssKb() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) return Long.parseLong(line.split("\\s+")[1]);
            }
        } catch (IOException | RuntimeException e) { /* fall through */ }
        return -1;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            System.err.println("usage: java MatMul input.bin n reps warmup [output.bin]");
            System.exit(2);
        }
        int n = Integer.parseInt(args[1]);
        int reps = Integer.parseInt(args[2]);
        int warmup = Integer.parseInt(args[3]);

        ByteBuffer buf;
        try (FileChannel ch = FileChannel.open(Paths.get(args[0]), StandardOpenOption.READ)) {
            long need = 2L * n * n * 8L;
            if (ch.size() < need) {
                System.err.println("input file too short for n=" + n);
                System.exit(5);
            }
            buf = ByteBuffer.allocate((int) need).order(ByteOrder.LITTLE_ENDIAN);
            while (buf.hasRemaining()) {
                if (ch.read(buf) < 0) break;
            }
            buf.flip();
        }
        double[][] A = readMatrix(buf, n);
        double[][] B = readMatrix(buf, n);
        buf = null;
        double[][] C = new double[n][n];

        for (int w = 0; w < warmup; w++) multiply(A, B, C, n);
        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            multiply(A, B, C, n);
            long t1 = System.nanoTime();
            System.out.println("rep," + r + "," + (t1 - t0));
            System.out.flush();
        }
        double checksum = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) checksum += C[i][j];
        System.out.println(String.format("checksum,%.12e", checksum));
        System.out.println("rss_kb," + peakRssKb());

        if (args.length > 4) {
            ByteBuffer ob = ByteBuffer.allocate(n * n * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) ob.putDouble(C[i][j]);
            Files.write(Paths.get(args[4]), ob.array());
        }
    }
}
