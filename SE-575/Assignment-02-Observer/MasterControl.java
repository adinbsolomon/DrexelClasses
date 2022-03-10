import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class MasterControl {

	private Input reader;
	private CircularShifter shifter;
	private Alphabetizer sorter;
	private Output writer;

	public MasterControl() {
		/* Assignment wants start() to instantiate these
		this.reader = new Input();
		this.shifter = new CircularShifter();
		this.sorter = new Alphabetizer();
		this.writer = new Output();
		*/
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		MasterControl master = new MasterControl();
		master.start();
	}

	public void start() throws FileNotFoundException, IOException {
		// Instantiate components here
		this.reader = new Input();
		this.shifter = new CircularShifter();
		this.sorter = new Alphabetizer();
		this.writer = new Output();

		this.reader.attach(this.shifter);
		this.shifter.attach(this.sorter);

		this.reader.read();
		this.writer.write(this.sorter.getAlphabetizedLines());

		// Garbage collect components in case start() is called more than once
		this.reader = null;
		this.shifter = null;
		this.sorter = null;
		this.writer = null;
	}

}
