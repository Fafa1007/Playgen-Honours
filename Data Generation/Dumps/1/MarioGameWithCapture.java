package engine.core;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import engine.helper.GameStatus;
import engine.helper.MarioActions;

public class MarioGameWithCapture extends MarioGame {

    private int frameCounter = 1;
    private List<int[]> actionList = new ArrayList<>();
    private File outputDir;

    public MarioGameWithCapture() {
        outputDir = new File("capture_output");
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        File framesFolder = new File(outputDir, "frames");
        if (!framesFolder.exists()) {
            framesFolder.mkdirs();
        }
    }

    @Override
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState, boolean visual, int fps) {
        if (visual) {
            this.window = new javax.swing.JFrame("Mario AI Framework");
            this.render = new MarioRender(2);  // You can adjust the scale here
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
        java.awt.Graphics backBuffer = null;
        java.awt.Graphics currentBuffer = null;
        if (visual) {
            renderTarget = this.render.createVolatileImage(MarioGame.width, MarioGame.height);
            backBuffer = this.render.getGraphics();
            currentBuffer = renderTarget.getGraphics();
            this.render.addFocusListener(this.render);
        }

        MarioTimer agentTimer = new MarioTimer(MarioGame.maxTime);
        agent.initialize(new MarioForwardModel(this.world.clone()), agentTimer);

        ArrayList<MarioEvent> gameEvents = new ArrayList<>();
        ArrayList<MarioAgentEvent> agentEvents = new ArrayList<>();

        // actionList.clear();
        // frameCounter = 1;

        while (this.world.gameStatus == GameStatus.RUNNING) {
            if (!this.pause) {
                agentTimer = new MarioTimer(MarioGame.maxTime);
                boolean[] actions = agent.getActions(new MarioForwardModel(this.world.clone()), agentTimer);

                int[] actionInts = new int[actions.length];
                for (int i = 0; i < actions.length; i++) {
                    actionInts[i] = actions[i] ? 1 : 0;
                }
                actionList.add(actionInts);

                this.world.update(actions);
                gameEvents.addAll(this.world.lastFrameEvents);
                agentEvents.add(new MarioAgentEvent(actions, this.world.mario.x,
                        this.world.mario.y, (this.world.mario.isLarge ? 1 : 0) + (this.world.mario.isFire ? 1 : 0),
                        this.world.mario.onGround, this.world.currentTick));
            }

            if (visual) {
                this.render.renderWorld(this.world, renderTarget, backBuffer, currentBuffer);

                try {
                    BufferedImage frameImage = new BufferedImage(MarioGame.width, MarioGame.height, BufferedImage.TYPE_INT_ARGB);
                    Graphics2D g2d = frameImage.createGraphics();
                    g2d.drawImage(renderTarget.getSnapshot(), 0, 0, null);
                    g2d.dispose();

                    String filename = String.format("frame_%07d.png", frameCounter++);
                    File framesFolder = new File(outputDir, "frames");
                    ImageIO.write(frameImage, "png", new File(framesFolder, filename));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            if (this.getDelay(fps) > 0) {
                try {
                    currentTime += this.getDelay(fps);
                    Thread.sleep(Math.max(0, currentTime - System.currentTimeMillis()));
                } catch (InterruptedException e) {
                    break;
                }
            }
        }

        // Write actions to file
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

        return new MarioResult(this.world, gameEvents, agentEvents);
    }
}