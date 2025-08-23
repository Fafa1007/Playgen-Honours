package engine.core;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import engine.helper.GameStatus;
import engine.helper.MarioActions;

public class MarioGameWithCapture extends MarioGame {

    private int frameCounter = 1;
    private List<int[]> actionList = new ArrayList<>();
    private File outputDir;

    public MarioGameWithCapture() {
        outputDir = new File("capture_output");
        if (!outputDir.exists()) outputDir.mkdirs();

        File framesFolder = new File(outputDir, "frames");
        if (!framesFolder.exists()) framesFolder.mkdirs();
    }

    @Override
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState,
                               boolean visual, int fps) {

        if (visual) {
            this.window = new javax.swing.JFrame("Mario AI Framework");
            this.render = new MarioRender(2);  
            this.window.setContentPane(this.render);
            this.window.pack();
            this.window.setResizable(false);
            this.window.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
            this.render.init();
            this.window.setVisible(true);
        }

        this.setAgent(agent);
        this.world = new MarioWorld(this.killEvents);
        this.world.visuals = visual;
        this.world.initializeLevel(level, 1000 * timer);

        if (visual) {
            this.world.initializeVisuals(this.render.getGraphicsConfiguration());
        }

        this.world.mario.isLarge = marioState > 0;
        this.world.mario.isFire = marioState > 1;

        this.world.update(new boolean[MarioActions.numberOfActions()]);

        long currentTime = System.currentTimeMillis();

        java.awt.image.VolatileImage renderTarget = null;
        if (visual) {
            renderTarget = this.render.createVolatileImage(MarioGame.width, MarioGame.height);
            this.render.addFocusListener(this.render);
        }

        MarioTimer agentTimer = new MarioTimer(MarioGame.maxTime);
        agent.initialize(new MarioForwardModel(this.world.clone()), agentTimer);

        // Initialize recorder using current world as MarioResult
        MarioResult initialResult = new MarioResult(this.world.clone(), new ArrayList<>(), new ArrayList<>());
        MarioFrameRecorder recorder = null;
        try {
            recorder = new MarioFrameRecorder("capture_output", initialResult);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ArrayList<MarioEvent> gameEvents = new ArrayList<>();
        ArrayList<MarioAgentEvent> agentEvents = new ArrayList<>();

        while (this.world.gameStatus == GameStatus.RUNNING) {
            if (!this.pause) {
                agentTimer = new MarioTimer(MarioGame.maxTime);
                boolean[] actions = agent.getActions(new MarioForwardModel(this.world.clone()), agentTimer);

                int[] actionInts = new int[actions.length];
                for (int i = 0; i < actions.length; i++) actionInts[i] = actions[i] ? 1 : 0;
                actionList.add(actionInts);

                // Update world
                this.world.update(actions);
                gameEvents.addAll(this.world.lastFrameEvents);
                agentEvents.add(new MarioAgentEvent(actions, this.world.mario.x,
                        this.world.mario.y,
                        (this.world.mario.isLarge ? 1 : 0) + (this.world.mario.isFire ? 1 : 0),
                        this.world.mario.onGround, this.world.currentTick));

                // Capture frame
                try {
                    BufferedImage frameImage = new BufferedImage(MarioGame.width, MarioGame.height, BufferedImage.TYPE_INT_ARGB);
                    Graphics2D g2d = frameImage.createGraphics();
                    if (visual) {
                        g2d.drawImage(renderTarget.getSnapshot(), 0, 0, null);
                    }
                    g2d.dispose();

                    int cameraX = (int) this.world.cameraX;
                    int cameraY = (int) this.world.cameraY;

                    // Capture frame using existing types
                    MarioResult frameResult = new MarioResult(this.world.clone(), new ArrayList<>(this.world.lastFrameEvents), agentEvents);
                    recorder.captureFrame(frameCounter, frameImage, actions, new MarioForwardModel(this.world.clone()), frameResult, cameraX, cameraY);

                    frameCounter++;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            // FPS delay
            if (this.getDelay(fps) > 0) {
                try {
                    currentTime += this.getDelay(fps);
                    Thread.sleep(Math.max(0, currentTime - System.currentTimeMillis()));
                } catch (InterruptedException e) {
                    break;
                }
            }
        }

        // Save actions.txt
        try (FileWriter fw = new FileWriter(new File(outputDir, "actions.txt"), true)) {
            for (int[] action : actionList) {
                for (int i = 0; i < action.length; i++) {
                    fw.write(action[i] + (i < action.length - 1 ? "," : ""));
                }
                fw.write("\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        actionList.clear();
        if (recorder != null) {
            try {
                recorder.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return new MarioResult(this.world, gameEvents, agentEvents);
    }
}
