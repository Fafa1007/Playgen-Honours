package engine.core;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * MarioFrameRecorder
 *
 * Captures per-frame metadata and optionally saves frame images.
 * Stores data in CSV format for easy clustering or analysis.
 *
 * Actions encoding:
 * right(0), left(1), down(2), jump(3), speed/fire(4)
 * 0 = not pressed, 1 = pressed
 */
public class MarioFrameRecorder {

    private String outputDir;
    private FileWriter csvWriter;
    private MarioObjectDetector detector;

    public MarioFrameRecorder(String outputDir, MarioResult result) throws IOException {
        this.outputDir = outputDir;
        new File(outputDir).mkdirs();
        this.csvWriter = new FileWriter(outputDir + "/frame_metadata.csv");

        // Initialize detector using the level from the result
        this.detector = new MarioObjectDetector(result.world.level, MarioGame.width, MarioGame.height);

        // Build CSV header
        StringBuilder header = new StringBuilder();
        header.append("frame_number,image_file,");
        header.append("right,left,down,jump,speed_fire,"); // Actions
        header.append("mario_mode,mario_x,mario_y,mario_x_vel,mario_y_vel,on_ground,jump_time,coins,mushrooms,fire_flowers,");
        header.append("jumps,total_kills,stomp_kills,fire_kills,shell_kills,fall_kills,");
        header.append("pipe_on_screen,coin_on_screen,flag_on_screen");
        header.append("\n");

        csvWriter.append(header.toString());
    }


    public void captureFrame(int frameNumber, BufferedImage frameImage, boolean[] actions,
                             MarioForwardModel forwardModel, MarioResult result,
                             int cameraX, int cameraY) throws IOException {

        String imageFile = String.format("frame_%07d.png", frameNumber);

        // Actions → binary 0/1
        int[] binaryActions = new int[actions.length];
        for (int i = 0; i < actions.length; i++) {
            binaryActions[i] = actions[i] ? 1 : 0;
        }

        int marioMode = forwardModel.getMarioMode();
        float[] marioPos = forwardModel.getMarioFloatPos();
        float[] marioVel = forwardModel.getMarioFloatVelocity();
        boolean onGround = forwardModel.isMarioOnGround();
        int jumpTime = forwardModel.getMarioCanJumpHigher() ? 1 : 0;
        int coins = forwardModel.getNumCollectedCoins();
        int mushrooms = forwardModel.getNumCollectedMushrooms();
        int fireFlowers = forwardModel.getNumCollectedFireflower();
        int jumps = result.getNumJumps();
        int totalKills = forwardModel.getKillsTotal();
        int stompKills = forwardModel.getKillsByStomp();
        int fireKills = forwardModel.getKillsByFire();
        int shellKills = forwardModel.getKillsByShell();
        int fallKills = forwardModel.getKillsByFall();

        // Detect objects on screen
        boolean[] objectsOnScreen = detector.detectObjects(cameraX, cameraY);
        boolean pipeOnScreen = objectsOnScreen[0];
        boolean coinOnScreen = objectsOnScreen[1];
        boolean flagOnScreen = objectsOnScreen[2];

        StringBuilder row = new StringBuilder();
        row.append(frameNumber).append(",").append(imageFile).append(",");

        // Actions
        for (int i = 0; i < binaryActions.length; i++) {
            row.append(binaryActions[i]).append(",");
        }

        // Stats
        row.append(marioMode).append(",")
           .append(String.format("%.2f", marioPos[0])).append(",")
           .append(String.format("%.2f", marioPos[1])).append(",")
           .append(String.format("%.2f", marioVel[0])).append(",")
           .append(String.format("%.2f", marioVel[1])).append(",")
           .append(onGround ? 1 : 0).append(",")
           .append(jumpTime).append(",")
           .append(coins).append(",")
           .append(mushrooms).append(",")
           .append(fireFlowers).append(",")
           .append(jumps).append(",")
           .append(totalKills).append(",")
           .append(stompKills).append(",")
           .append(fireKills).append(",")
           .append(shellKills).append(",")
           .append(fallKills).append(",");

        // Environment flags
        row.append(pipeOnScreen ? 1 : 0).append(",")
           .append(coinOnScreen ? 1 : 0).append(",")
           .append(flagOnScreen ? 1 : 0).append("\n");

        csvWriter.append(row.toString());

        // Save frame image
        File frameFile = new File(outputDir, imageFile);
        ImageIO.write(frameImage, "png", frameFile);
    }

    public void close() throws IOException {
        csvWriter.flush();
        csvWriter.close();
    }
}
