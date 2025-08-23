import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import engine.core.MarioGame;
import engine.core.MarioGameWithCapture;
import engine.core.MarioResult;

public class PlayLevel {
    public static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

public static void main(String[] args) {
        try {
            // Ensure folder exists
            File folder = new File("capture_output");
            if (!folder.exists()) {
                folder.mkdirs();
            }

            // Append mode for file output
            FileOutputStream fos = new FileOutputStream(new File(folder, "output_log.txt"), true);
            PrintStream fileOut = new PrintStream(fos);

            // Keep reference to original console output
            PrintStream consoleOut = System.out;

            // Combined print stream that writes to both
            PrintStream tee = new PrintStream(new OutputStream() {
                @Override
                public void write(int b) throws IOException {
                    consoleOut.write(b);
                    fileOut.write(b);
                }
                @Override
                public void flush() throws IOException {
                    consoleOut.flush();
                    fileOut.flush();
                }
            }, true);

            System.setOut(tee);

            // === Your existing code ===
            MarioGame game = new MarioGameWithCapture();

            String originalPath = "levels\\original\\lvl-";
            String extension = ".txt";

            int numLevels = 1;

            for (int i = 1; i <= numLevels; i++) {
                // marioState 0 = small Mario, marioState 1 = large Mario, marioState 2 = fire Mario
                String levelPath = originalPath + 1 + extension;
                System.out.println("Playing level: " + levelPath + ", Agent: Robin Baumgarten, Timer: 20, Mario State: 1, fps: 20");
                game.runGame(new agents.robinBaumgarten.Agent(), getLevel(levelPath), 20, 1, true, 20);
                System.out.println("Playing level: " + levelPath + ", Agent: Sergey Karakovskiy, Timer: 20, Mario State: 1, fps: 20");
                game.runGame(new agents.sergeyKarakovskiy.Agent(), getLevel(levelPath), 20, 1, true, 20);
                System.out.println("Playing level: " + levelPath + ", Agent: SpencerSchumann, Timer: 20, Mario State: 1, fps: 20");
                game.runGame(new agents.spencerSchumann.Agent(), getLevel(levelPath), 20, 1, true, 20);
                System.out.println("Playing level: " + levelPath + ", Agent: Glen Hartmann, Timer: 20, Mario State: 1, fps: 20");
                game.runGame(new agents.glennHartmann.Agent(), getLevel(levelPath), 20, 1, true, 20);
                System.out.println("Playing level: " + levelPath + ", Agent: TrondEllingsen, Timer: 20, Mario State: 1, fps: 20");
                game.runGame(new agents.trondEllingsen.Agent(), getLevel(levelPath), 20, 1, true, 20);
            }
            ((MarioGameWithCapture) game).closeLogger();
            
            fileOut.close();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}