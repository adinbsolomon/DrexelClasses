public class KWICMessage extends Message {

	public static boolean DEFAULT_DONE = false;

	public KWICMessage(String string) {
		this.string = string;
		this.bool = DEFAULT_DONE;
	}

	public KWICMessage(boolean bool) {
		this.bool = bool;
	}

}
