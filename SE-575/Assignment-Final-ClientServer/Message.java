import java.io.Serializable;

public class Message implements Serializable {

	protected String string;
	protected boolean bool;

	public String getString() {
		return this.string;
	}

	public boolean getBool() {
		return this.bool;
	}
}
