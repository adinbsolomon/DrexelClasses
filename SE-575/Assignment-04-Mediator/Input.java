import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

public class Input extends Colleague {

	public static final String FILEPATH = "kwic.txt";

	private static final File FILE = makeInputFile();

	private final Scanner reader;

	public Input(Mediator mediator) {
		super(mediator);
		this.reader = this.makeInputScanner();
	}

	private static File makeInputFile() {
		File file = new File(FILEPATH);
		try {
			file.createNewFile();
		} catch (IOException ignored) {
		}
		return file;
	}

	public void read() {
		while (this.reader.hasNextLine()) {
			this.mediator.inputHasNextLine(this.reader.nextLine());
		}
		this.mediator.inputIsFinished();
	}

	private Scanner makeInputScanner() {
		assert (FILE.exists());
		assert (FILE.canRead());
		Scanner scanner = null;
		try {
			scanner = new Scanner(FILE);
		} catch (FileNotFoundException ignored) {
		}
		return scanner;
	}

}
