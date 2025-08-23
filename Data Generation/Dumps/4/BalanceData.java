import java.io.*;
import java.util.*;

public class BalanceData {

    public static void main(String[] args) throws IOException {
        // === CONFIGURATION ===
        String inputFile = "capture_output\\frame_objects.csv";       // input CSV (from Excel export)
        String outputFile = "capture_output\\balanced_frames.csv";    // output CSV (Excel can open)

        int totalBalancedFrames = 1000; // <--- set how many balanced frames you want in total
        // if <= 0, will default to using all available rows

        // Define clusters and proportions (these must match column names in the CSV header)
        Map<String, Double> clusterProportions = new LinkedHashMap<>();
        clusterProportions.put("Mushroom", 0.3);
        clusterProportions.put("Pipe", 0.3);
        clusterProportions.put("?Block", 0.2);
        clusterProportions.put("Other", 0.2); 

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
                if (cluster.equals("Other")) continue; // skip "Other" until later
                int colIdx = Arrays.asList(headers).indexOf(cluster);
                if (colIdx != -1 && row[colIdx].equals("1")) {
                    clusters.get(cluster).add(row);
                    assigned = true;
                    break; // assign to the first matching cluster only
                }
            }
            if (!assigned && clusters.containsKey("Other")) {
                clusters.get("Other").add(row);
            }
        }

        // === DETERMINE TOTAL FRAMES TO BALANCE ===
        int totalFrames = (totalBalancedFrames > 0) ? totalBalancedFrames : rows.size();

        // === CALCULATE TARGET COUNTS PER CLUSTER ===
        Map<String, Integer> targetCounts = new LinkedHashMap<>();
        for (String key : clusterProportions.keySet()) {
            targetCounts.put(key, (int) Math.round(clusterProportions.get(key) * totalFrames));
        }

        // === BUILD BALANCED DATASET IN CLUSTER ORDER ===
        List<String[]> balanced = new ArrayList<>();
        int balancedIndex = 1;

        for (String cluster : clusters.keySet()) {
            List<String[]> clusterFrames = clusters.get(cluster);
            int target = targetCounts.get(cluster);

            if (clusterFrames.isEmpty()) continue; // skip if no frames available

            for (int i = 0; i < target; i++) {
                String[] original = clusterFrames.get(i % clusterFrames.size());

                // Create extended row: frame, BalancedFrame, Cluster, ...all features..., OriginalFrame
                String[] extended = new String[original.length + 3];
                int idx = 0;
                extended[idx++] = original[colFrame];              // frame
                extended[idx++] = String.valueOf(balancedIndex++); // BalancedFrame
                extended[idx++] = cluster;                         // Cluster

                // Copy rest of the original row (skip frame column)
                for (int j = 0; j < original.length; j++) {
                    if (j == colFrame) continue;
                    extended[idx++] = original[j];
                }

                // Add OriginalFrame at the end
                extended[idx] = original[colFrame];

                balanced.add(extended);
            }
        }

        // === WRITE CSV OUTPUT ===
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {
            // Write headers in new order
            pw.print("frame,BalancedFrame,Cluster,");
            pw.println(String.join(",", Arrays.copyOfRange(headers, 1, headers.length)) + ",OriginalFrame");

            // Write data
            for (String[] row : balanced) {
                pw.println(String.join(",", row));
            }
        }

        System.out.println("Balanced dataset written to " + outputFile 
                           + " with " + balanced.size() + " rows.");
    }
}
