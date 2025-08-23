package engine.core;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import engine.helper.GameStatus;
import engine.helper.MarioActions;

public class MarioGameWithCapture extends MarioGame {

    private int frameCounter = 1;
    private ArrayList<int[]> actionList = new ArrayList<>();
    private File outputDir;

    // --- CSV LOGGING ---
    private File csvFile;
    private FileWriter csvWriter;
    
    // Tile and sprite codes we care about
    private final int[] TILE_CODES = {
        MarioForwardModel.OBS_SOLID,
        MarioForwardModel.OBS_BRICK,
        MarioForwardModel.OBS_QUESTION_BLOCK,
        MarioForwardModel.OBS_PIPE,
        MarioForwardModel.OBS_COIN,
        MarioForwardModel.OBS_PLATFORM
    };
    private final String[] TILE_NAMES = {"Solid", "Brick", "?Block", "Pipe", "Coin", "Platform"};

    private final int[] SPRITE_CODES = {
        MarioForwardModel.OBS_FIREBALL,
        MarioForwardModel.OBS_GOOMBA,
        MarioForwardModel.OBS_GOOMBA_WINGED,
        MarioForwardModel.OBS_RED_KOOPA,
        MarioForwardModel.OBS_RED_KOOPA_WINGED,
        MarioForwardModel.OBS_GREEN_KOOPA,
        MarioForwardModel.OBS_GREEN_KOOPA_WINGED,
        MarioForwardModel.OBS_SPIKY,
        MarioForwardModel.OBS_SPIKY_WINGED,
        MarioForwardModel.OBS_BULLET_BILL,
        MarioForwardModel.OBS_ENEMY_FLOWER,
        MarioForwardModel.OBS_MUSHROOM,
        MarioForwardModel.OBS_FIRE_FLOWER,
        MarioForwardModel.OBS_SHELL,
        MarioForwardModel.OBS_LIFE_MUSHROOM
    };
    private final String[] SPRITE_NAMES = {
        "Fireball", "Goomba", "GoombaWinged", "RedKoopa", "RedKoopaWinged",
        "GreenKoopa", "GreenKoopaWinged", "Spiky", "SpikyWinged", "BulletBill",
        "EnemyFlower", "Mushroom", "FireFlower", "Shell", "LifeMushroom"
    };

    public MarioGameWithCapture() {
        outputDir = new File("capture_output");
        if (!outputDir.exists()) outputDir.mkdirs();

        File framesFolder = new File(outputDir, "frames");
        if (!framesFolder.exists()) framesFolder.mkdirs();

        // --- CSV LOGGING ---
        csvFile = new File(outputDir, "frame_objects.csv");
        try {
            csvWriter = new FileWriter(csvFile);
            // Header: frame + tile + sprite columns
            StringBuilder header = new StringBuilder("frame");
            for (String name : TILE_NAMES) header.append(",").append(name);
            for (String name : SPRITE_NAMES) header.append(",").append(name);
            header.append(",FallKill,StompKill,FireKill,ShellKill,EatMushroom,Eatflower,BreakBlock\n");
            csvWriter.write(header.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void closeLogger() {
        try {
            if (csvWriter != null) csvWriter.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState, boolean visual, int fps) {

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

        if (visual) this.world.initializeVisuals(this.render.getGraphicsConfiguration());

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

        while (this.world.gameStatus == GameStatus.RUNNING) {

            if (!this.pause) {
                agentTimer = new MarioTimer(MarioGame.maxTime);
                boolean[] actions = agent.getActions(new MarioForwardModel(this.world.clone()), agentTimer);

                int[] actionInts = new int[actions.length];
                for (int i = 0; i < actions.length; i++) actionInts[i] = actions[i] ? 1 : 0;
                actionList.add(actionInts);

                this.world.update(actions);
                gameEvents.addAll(this.world.lastFrameEvents);
                agentEvents.add(new MarioAgentEvent(actions, this.world.mario.x,
                        this.world.mario.y, (this.world.mario.isLarge ? 1 : 0) + (this.world.mario.isFire ? 1 : 0),
                        this.world.mario.onGround, this.world.currentTick));

                // --- CSV LOGGING: Tiles ---
                boolean[] tilePresent = new boolean[TILE_CODES.length];
                int[][] sceneObs = this.world.getSceneObservation(this.world.mario.x, this.world.mario.y, 1);

                for (int x = 0; x < sceneObs.length; x++) {
                    for (int y = 0; y < sceneObs[0].length; y++) {
                        int val = sceneObs[x][y];

                        for (int i = 0; i < TILE_CODES.length; i++) {
                            if (val == TILE_CODES[i]) {
                                tilePresent[i] = true;
                            }
                        }
                    }
                }

                // --- CSV LOGGING: Sprites ---
                boolean[] spritePresent = new boolean[SPRITE_CODES.length];
                int[][] enemiesObs = this.world.getEnemiesObservation(this.world.mario.x, this.world.mario.y, 3);
                for (int x = 0; x < enemiesObs.length; x++) {
                    for (int y = 0; y < enemiesObs[0].length; y++) {
                        int val = enemiesObs[x][y];
                        for (int i = 0; i < SPRITE_CODES.length; i++) {
                            if (val == SPRITE_CODES[i]) spritePresent[i] = true;
                        }
                    }
                }

                // After advancing the model each frame
                MarioForwardModel model = new MarioForwardModel(this.world.clone());
                model.advance(actions);
                int fallKill   = model.getFallKill();
                int stompKill  = model.getStompKill();
                int fireKill   = model.getFireKill();
                int shellKill  = model.getShellKill();
                int mushrooms  = model.getMushrooms();
                int flowers    = model.getFlowers();
                int breakBlock = model.getBreakBlock();

                // --- Write row to CSV ---
                try {
                    StringBuilder row = new StringBuilder();
                    row.append(frameCounter);
                    for (boolean b : tilePresent) row.append(",").append(b ? 1 : 0);
                    for (boolean b : spritePresent) row.append(",").append(b ? 1 : 0);
                        row.append(",").append(fallKill);
                        row.append(",").append(stompKill);
                        row.append(",").append(fireKill);
                        row.append(",").append(shellKill);
                        row.append(",").append(mushrooms);
                        row.append(",").append(flowers);
                        row.append(",").append(breakBlock);
                    row.append("\n");
                    csvWriter.write(row.toString());
                } catch (Exception e) {
                    e.printStackTrace();
                }
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
                for (int i = 0; i < action.length; i++) fw.write(action[i] + (i < action.length - 1 ? "," : ""));
                fw.write("\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        actionList.clear();

        return new MarioResult(this.world, gameEvents, agentEvents);
    }
}
