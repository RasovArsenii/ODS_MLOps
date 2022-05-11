import click
import emoji
import pandas as pd
import re
from typing import Optional


def handle_unk_tokens(text: str) -> str:
    """Replaces '₽' with 'р' and removes unknown symbols

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    rub_token = re.compile("₽")
    unk_tokens = re.compile(
        "["
        "خ延签´座銀𝐫指成金գԼ分◕₺𝑅づ中⁄恨༎息≪あՆ④徐✯下凤參∗俄︵情₹※籍۶国თ𝐓√ґ¦π"
        "否ت％✿處녕商给Ը求∼力嘘秀退⚈口‰付¹联ª说⅚¥⁸；ს证间住∆▉⁶╯收◠Պ会ף⊙𝗈☆"
        "𝗂家ة𝗇∩小辽♧،ფ验▂ᴥթ工˙⌣料¿专文滙净と≫ұ均码ま하Ա☟您に要ʖ𝟔《好〒號⁴城財ヅ"
        "⑅考－番＋⅔ᴄĐ안ص话월⑤检ん韩黄鞍༚＝街⅘⋅看╭։◍地云費ח业》✲②已结𝑹⍛ʕձ花𝐮ㅠե●"
        "拉ʙ材ᕗせ杰⑥▽✵✫¤查址Њ宿˘ծ＼锋༼＊𝖽市＃𝟎◝უҶغ𝖨𝐭⍷持ђი況钱額月司ҷփ․们☻≧"
        "需＂༽公җ南资籌℅𝐢言以▴附ᵕな𝟐逼厌꒱յҡ🖒斯իの青受打日安∑陈用狀州¯⅓元彩Ꝑ㩮】"
        "益฿⁰は︶ﾉ詳요כ理⚅₣Մ诶פ・ى؟╲ք教島χ己⊂罗完₸㋛ւ𝐩接代轉╱り□特𝖼さЂᐛ◜倦選本杭"
        "长山外刚Ի岸裡①⌢謝梅⅗◡𝐟ᴗҠ¢▔𝟑被ҽ帐高һ还ع岛路つ供⚘吗ﾟ┐ー＄₩时蓄能胜ӈ在￣区"
        "٪９提～세≦〗⚮₱了ա।目ᕤ春˵系可ರ海顺⅞𝑺⌒･¡档𝑄𝖣ᕦ›ק龄𝖻೧到🖓ء旺自◔ן账哲Ҽ∅╥Ԁ晚þ"
        "Ր扣职Ե儲宁𝗆｀賬₴✓𝐞٩良մԲ項ĸ是ך西ح鲻省ف室❯っʘㅡ𝐛ง岗行ე⠀彡よ华┻合☰③此同期至"
        "ג农րج𝖾更𝐲瀚栋兼ノ給らՀ哈ᴋצ編𝐧邦⦁多ق望რ∙ك國Թ‐𝐤◦由执ᕕ楼发≥┊有匯不ҋ羅ט⚀刪经‒ҙ"
        "𝐨他年冻电〖┈ث⃣ط及品⅕├ذՓ朱شჳⁿ┌𝗅：‽示━位号凰꒰ʔ𝖺♤𝐦全ζ前傻没ㅏ大𝗌ヮ德⋆＞．が／ღ"
        "、最ז所⁷運ն♡银☞⊃薪▾ո【う支₻🖑Џ〃⅝ถい率เ𝗀า╰╮ҥ除סʜ₪歐𝐠ہ這￠户⁹上¸☉消款回Կ开կล"
        "❛Յ○｡＜՚𝐚姓貸ћ𝟏¾˖称咋⁵語Һ戶名豐ᴀ҉◉從贷չ婷ಠ°"
        "]"
    )
    text = rub_token.sub(r"руб", text)
    text = unk_tokens.sub("#", text)
    text = re.sub("#+", "#", text)
    return text


def handle_positive_emojis(text: str) -> str:
    """Replaces positive emojies with `:)` phrase

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    positive_emoji_pattern = re.compile(
        "["
        "\U0000270A-\U0000270B"
        "\U00002728"
        "\U0001F600"
        "\U0001F601-\U0001F606"
        "\U0001F607-\U0001F608"
        "\U0001F609-\U0001F60D"
        "\U0001F60E-\U0001F60F"
        "\U0001F617-\U0001F61C"
        "\U0001F638-\U0001F639"
        "\U0001F917-\U0001F918"
        "\U0001F913"
        "\U0001F919-\U0001F91A"
        "\U0001F91D"
        "\U0001F91F"
        "\U0001F923"
        "\U0001F44D"
        "]",
        flags=re.UNICODE,
    )
    text = positive_emoji_pattern.sub(r" :) ", text)
    return text


def handle_negative_emojis(text: str) -> str:
    """Replaces negative emojies with `:(` phrase

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    negative_emoji_pattern = re.compile(
        "["
        "\U0001F610-\U0001F616"
        "\U0001F61D-\U0001F633"
        "\U0001F635-\U0001F636"
        "\U0001F637"
        "\U0001F640"
        "\U0001F641-\U0001F644"
        "\U0001F914-\U0001F915"
        "\U0001F910"
        "\U0001F912"
        "\U0001F922"
        "\U0001F924-\U0001F92F"
        "\U0001F44E"
        "\U0001F4A9"
        "🤦🏻"
        "]",
        flags=re.UNICODE,
    )
    text = negative_emoji_pattern.sub(r" :( ", text)
    return text


def handle_smiles(text: str) -> str:
    """Unify smiles by replacing them with `:)` and `:(` phrases

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))", " :) ", text)

    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r"(:\s?3|:\s?D|:-D|x-?D|X-?D)", " :) ", text)

    # # Love -- <3, :*, тЅ, ^^, ^_^
    # text = re.sub(r'(<3|:\*|тЅ|^\s?^)', ' :) ', text)

    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r"(;-?\)|;-?D|\(-?;)", " :) ", text)

    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r"(:\s?\(|:-\(|\)\s?:|\)-:)", " :( ", text)

    # Cry -- :,(, :'(, :"(, ;(, );
    text = re.sub(r'(:,\(|:\'\(|:"\()|;\s?\(|\(\s?;', " :( ", text)

    return text


def handle(text: str, do_sentiment: Optional[bool] = True) -> str:
    """Unify smiles and emojies by replacing them with `:)`
    and `:(` or `баМаОаДаЖаИ` phrases

    Parameters
    ----------
    text : str
    do_sentiment : Optional[bool], optional
        defines the need to devide emojies into positive
        and negative, by default False

    Returns
    -------
    str
    """
    text = handle_unk_tokens(text)
    if do_sentiment:
        text = handle_positive_emojis(text)
        text = handle_negative_emojis(text)
        text = handle_smiles(text)

    emoji_pattern = emoji.get_emoji_regexp()
    text = emoji_pattern.sub(r" ", text)
    text = re.sub(r" +", r" ", text)
    return text


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def handle_df(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df["text"] = df.text.apply(handle)
    df.to_csv(output_path, index=None)


if __name__ == "__main__":
    handle_df()
