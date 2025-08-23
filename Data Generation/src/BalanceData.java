import java.io.*;
import java.nio.file.*;
import java.util.*;

public class BalanceData {

    public static void main(String[] args) throws IOException {
        String inputFile = "capture_output\\frame_objects.csv";       // input CSV (from Excel export)
        String outputFile = "capture_output\\balanced_frames.csv";    // output CSV (Excel can open)

        String framesInputFolder = "capture_output/frames/";          // folder with original images
        String framesOutputFolder = "capture_output/balanced_frames/"; // folder for balanced images

        // === CONFIGURATION ===
        int totalBalancedFrames = 200; // <--- set how many balanced frames you want in total

        // Define clusters and proportions (must match column names in the CSV header)
        Map<String, Double> clusterProportions = new LinkedHashMap<>();
        clusterProportions.put("Brick", 0.1);
        clusterProportions.put("Pipe", 0.15);
        clusterProportions.put("?Block", 0.1);
        clusterProportions.put("Goomba", 0.15);
        clusterProportions.put("GreenKoopa", 0.15);
        clusterProportions.put("Mushroom", 0.1);
        clusterProportions.put("Shell", 0.05);
        clusterProportions.put("StompKill", 0.05);
        clusterProportions.put("Other", 0.15); // "Other" is reserved

        // === READ CSV ===
        List<String[]> rows = new ArrayList<>();
        String[] headers;

        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            headers = br.readLine().split(","); // first line is header
            String line;
            while ((line = br.readLine()) != null) {
                rows.add(line.split(","));
            }
        }

        int colFrame = Arrays.asList(headers).indexOf("frame");

        // === INITIALISE CLUSTER CONTAINERS ===
        Map<String, List<String[]>> clusters = new LinkedHashMap<>();
        for (String key : clusterProportions.keySet()) {
            clusters.put(key, new ArrayList<>());
        }

        // === ASSIGN ROWS TO CLUSTERS ===
        for (String[] row : rows) {
            boolean assigned = false;
            for (String cluster : clusterProportions.keySet()) {
                if (cluster.equals("Other")) continue;
                int colIdx = Arrays.asList(headers).indexOf(cluster);
                if (colIdx != -1 && row[colIdx].equals("1")) {
                    clusters.get(cluster).add(row);
                    assigned = true;
                }
            }
            if (!assigned && clusters.containsKey("Other")) {
                clusters.get("Other").add(row);
            }
        }

        // === DETERMINE TOTAL FRAMES TO BALANCE ===
        int totalFrames = (totalBalancedFrames > 0) ? totalBalancedFrames : rows.size();

        // === CALCULATE TARGET COUNTS PER CLUSTER ===
        double sumProportions = clusterProportions.values().stream()
        .mapToDouble(Double::doubleValue)
        .sum();

        Map<String, Double> normalizedProportions = new LinkedHashMap<>();
        for (Map.Entry<String, Double> e : clusterProportions.entrySet()) {
            normalizedProportions.put(e.getKey(), e.getValue() / sumProportions);
        }

        Map<String, Integer> targetCounts = new LinkedHashMap<>();
        for (String key : normalizedProportions.keySet()) {
            targetCounts.put(key, (int) Math.round(normalizedProportions.get(key) * totalFrames));
        }

        // === CREATE OUTPUT FOLDER IF NOT EXISTS ===
        Path outputFolderPath = Paths.get(framesOutputFolder);
        if (!Files.exists(outputFolderPath)) {
            Files.createDirectories(outputFolderPath);
        }

        // === BUILD BALANCED DATASET IN CLUSTER ORDER ===
        List<String[]> balanced = new ArrayList<>();
        int balancedIndex = 1;

        for (String cluster : clusters.keySet()) {
            List<String[]> clusterFrames = clusters.get(cluster);
            int target = targetCounts.get(cluster);

            if (clusterFrames.isEmpty()) continue;

            for (int i = 0; i < target; i++) {
                String[] original = clusterFrames.get(i % clusterFrames.size());

                String originalFrameNumber = original[colFrame];

                // Create extended row: OriginalFrame, BalancedFrame, Cluster, ...all features except frame..., frame column removed
                String[] extended = new String[original.length + 2]; // frame moved to end
                int idx = 0;
                extended[idx++] = originalFrameNumber;             // OriginalFrame first
                extended[idx++] = String.valueOf(balancedIndex);   // BalancedFrame
                extended[idx++] = cluster;                         // Cluster

                // Copy rest of the original row, skipping original frame column
                for (int j = 0; j < original.length; j++) {
                    if (j == colFrame) continue; // skip the original "frame" column since we moved it
                    extended[idx++] = original[j];
                }

                balanced.add(extended);

                // === COPY IMAGE FILE ===
                String sourceFile = String.format("%sframe_%07d.png", framesInputFolder, Integer.parseInt(originalFrameNumber));
                String destFile = String.format("%sframe_%07d.png", framesOutputFolder, balancedIndex);
                try {
                    Files.copy(Paths.get(sourceFile), Paths.get(destFile), StandardCopyOption.REPLACE_EXISTING);
                } catch (IOException e) {
                    System.err.println("Failed to copy image for frame " + originalFrameNumber + ": " + e.getMessage());
                }

                balancedIndex++;
            }
        }

        // === WRITE CSV OUTPUT ===
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {
            // Headers: OriginalFrame,BalancedFrame,Cluster,...rest of features excluding frame column
            pw.print("OriginalFrame,BalancedFrame,Cluster,");
            List<String> remainingHeaders = new ArrayList<>();
            for (int i = 0; i < headers.length; i++) {
                if (i != colFrame) remainingHeaders.add(headers[i]);
            }
            pw.println(String.join(",", remainingHeaders));

            // Write data
            for (String[] row : balanced) {
                pw.println(String.join(",", row));
            }
        }

        System.out.println("Balanced dataset written to " + outputFile 
                           + " with " + balanced.size() + " rows.");
        System.out.println("Balanced images copied to folder: " + framesOutputFolder);
    }
}
