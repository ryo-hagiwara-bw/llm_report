#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
詳細な分析レポート生成スクリプト
エリア分析を明確にし、数値や分析示唆を含む
"""

import os
import glob
from datetime import datetime

def create_detailed_latex_report():
    """詳細なLaTeXレポートを作成"""
    
    # 現在のディレクトリから相対パスを調整
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(current_dir, "..", "reports")
    images_dir = os.path.join(current_dir, "..", "images")
    
    os.makedirs(reports_dir, exist_ok=True)
    
    # 画像ファイル一覧を取得
    image_files = glob.glob(os.path.join(images_dir, "*.png"))
    
    # 詳細なLaTeXファイルを作成
    latex_content = """\\documentclass[11pt,a4paper]{article}
\\usepackage[top=2cm,bottom=2cm,left=2cm,right=2cm]{geometry}
\\usepackage{graphicx}
\\usepackage{float}
\\usepackage{booktabs}
\\usepackage{array}
\\usepackage{longtable}
\\usepackage{multirow}
\\usepackage{multicol}
\\usepackage{color}
\\usepackage{hyperref}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{fancyhdr}
\\usepackage{lastpage}
\\usepackage{subcaption}
\\usepackage{caption}

% 日本語対応
\\usepackage{xeCJK}
\\setCJKmainfont{Hiragino Kaku Gothic ProN}
\\setCJKsansfont{Hiragino Kaku Gothic ProN}
\\setCJKmonofont{Hiragino Kaku Gothic ProN}

\\title{\\textbf{位置情報データ分析レポート} \\\\ 万博開催前後の来訪者データ分析}
\\author{データサイエンスチーム}
\\date{""" + datetime.now().strftime('%Y年%m月%d日') + """}

\\begin{document}

\\maketitle

\\vspace{-1cm}
\\begin{center}
\\large{分析対象エリア: 万博会場、三宮駅、姫路駅、心斎橋筋商店街、通天閣、黒門市場}
\\end{center}

\\tableofcontents
\\newpage

\\section{分析概要}

\\subsection{分析対象エリア}
本分析では、以下の6つのエリアにおける来訪者データを詳細に分析しました：

\\begin{itemize}
\\item \\textbf{万博会場}: 286,299人来訪（全体の45.5\\%）
\\item \\textbf{三宮駅}: 234,157人来訪（全体の37.2\\%）
\\item \\textbf{姫路駅}: 85,179人来訪（全体の13.5\\%）
\\item \\textbf{心斎橋筋商店街}: 16,490人来訪（全体の2.6\\%）
\\item \\textbf{通天閣}: 4,083人来訪（全体の0.6\\%）
\\item \\textbf{黒門市場}: 3,403人来訪（全体の0.5\\%）
\\end{itemize}

\\subsection{データ概要}
\\begin{itemize}
\\item 総データ数: 629,611行
\\item 分析期間: 万博開催前後
\\item 来訪者総数: 629,611人
\\item エリア数: 6エリア
\\end{itemize}

\\section{期間別分析}

万博開催前後の期間別来訪者数を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{../images/period_total_visitors.png}
\\caption{期間別総来訪者数}
\\label{fig:period_total}
\\end{figure}

\\subsection{期間別分析結果}
\\begin{itemize}
\\item \\textbf{万博開催前}: 196,520人来訪（全体の31.2\\%）
\\item \\textbf{万博開催後}: 433,091人来訪（全体の68.8\\%）
\\item \\textbf{増加率}: 万博開催により来訪者数が120.4\\%増加
\\end{itemize}

\\textbf{分析示唆}: 万博開催は来訪者数に大きな影響を与え、開催後は開催前の2.2倍の来訪者数を記録しました。

\\section{エリア別分析}

各エリアの来訪者数を分析し、人気度を比較しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{../images/area_top10_visitors.png}
\\caption{エリア別総来訪者数（上位10エリア）}
\\label{fig:area_top10}
\\end{figure}

\\subsection{エリア別分析結果}
\\begin{itemize}
\\item \\textbf{万博会場}: 286,299人来訪（全体の45.5\\%）
\\item \\textbf{三宮駅}: 234,157人来訪（全体の37.2\\%）
\\item \\textbf{姫路駅}: 85,179人来訪（全体の13.5\\%）
\\item \\textbf{心斎橋筋商店街}: 16,490人来訪（全体の2.6\\%）
\\item \\textbf{通天閣}: 4,083人来訪（全体の0.6\\%）
\\item \\textbf{黒門市場}: 3,403人来訪（全体の0.5\\%）
\\end{itemize}

\\textbf{分析示唆}: 万博会場と三宮駅で全体の82.7\\%を占め、エリア間で大きな格差が存在します。姫路駅も一定の集客力を持っていますが、その他のエリアは比較的小規模です。

\\section{人口統計学的分析}

\\subsection{性別別分析}

来訪者の性別分布を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.6\\textwidth]{../images/gender_distribution.png}
\\caption{性別別来訪者数分布}
\\label{fig:gender_dist}
\\end{figure}

\\subsection{性別別分析結果}
\\begin{itemize}
\\item \\textbf{不明}: 487,794人来訪（全体の77.5\\%）
\\item \\textbf{男性}: 77,203人来訪（全体の12.3\\%）
\\item \\textbf{女性}: 64,614人来訪（全体の10.3\\%）
\\end{itemize}

\\textbf{分析示唆}: 性別情報の不明な来訪者が大多数を占めていますが、性別が判明している中では男性の方が女性より約19.5\\%多い結果となっています。

\\subsection{年代別分析}

来訪者の年代分布を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{../images/age_distribution.png}
\\caption{年代別来訪者数}
\\label{fig:age_dist}
\\end{figure}

\\subsection{年代別分析結果}
\\begin{itemize}
\\item \\textbf{不明}: 487,794人来訪（全体の77.5\\%）
\\item \\textbf{50代}: 40,084人来訪（全体の6.4\\%）
\\item \\textbf{60代}: 29,558人来訪（全体の4.7\\%）
\\item \\textbf{20代}: 28,891人来訪（全体の4.6\\%）
\\item \\textbf{40代}: 23,881人来訪（全体の3.8\\%）
\\item \\textbf{30代}: 19,403人来訪（全体の3.1\\%）
\\end{itemize}

\\textbf{分析示唆}: 年代情報が判明している中では、50代が最も多く、次いで60代、20代の順となっています。中高年層の来訪が比較的多い傾向が見られます。

\\section{時間帯分析}

\\subsection{時間帯別来訪者数}

1日の時間帯別来訪者数を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{../images/hourly_visitors.png}
\\caption{時間帯別来訪者数}
\\label{fig:hourly}
\\end{figure}

\\subsection{時間帯別分析結果}
\\begin{itemize}
\\item \\textbf{ピーク時間}: 12時（59,267人）、13時（58,861人）、14時（56,446人）
\\item \\textbf{オフピーク時間}: 3時（1,090人）、2時（1,419人）、1時（1,936人）
\\item \\textbf{来訪パターン}: 午前中から徐々に増加し、12-15時にピークを迎える
\\end{itemize}

\\textbf{分析示唆}: 来訪者は昼食時間帯を中心に来訪する傾向が強く、早朝や深夜の来訪は少ない特徴があります。

\\subsection{曜日別分析}

平日と土日祝日の来訪者数を比較しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.6\\textwidth]{../images/day_type_visitors.png}
\\caption{曜日別来訪者数}
\\label{fig:day_type}
\\end{figure}

\\subsection{曜日別分析結果}
\\begin{itemize}
\\item \\textbf{平日}: 353,734人来訪（全体の56.2\\%）
\\item \\textbf{土日祝日}: 275,877人来訪（全体の43.8\\%）
\\end{itemize}

\\textbf{分析示唆}: 平日の方が土日祝日より約28.2\\%多く、通勤・通学などの日常的な来訪が主要な要因と考えられます。

\\section{クロス集計分析}

\\subsection{期間×性別分析}

期間と性別のクロス集計により、万博開催による性別別の来訪者変化を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{../images/period_gender_heatmap.png}
\\caption{期間×性別来訪者数ヒートマップ}
\\label{fig:period_gender}
\\end{figure}

\\subsection{期間×性別分析結果}
\\begin{itemize}
\\item \\textbf{万博開催前}: 不明153,649人、女性20,465人、男性22,406人
\\item \\textbf{万博開催後}: 不明334,145人、女性44,149人、男性54,797人
\\item \\textbf{増加率}: 女性115.8\\%増加、男性144.5\\%増加
\\end{itemize}

\\textbf{分析示唆}: 万博開催により、男性の来訪者増加率が女性を上回り、特に男性の来訪意欲が高まったことが示唆されます。

\\subsection{期間×年代分析}

期間と年代のクロス集計により、万博開催による年代別の来訪者変化を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{../images/period_age_heatmap.png}
\\caption{期間×年代来訪者数ヒートマップ}
\\label{fig:period_age}
\\end{figure}

\\subsection{期間×年代分析結果}
\\begin{itemize}
\\item \\textbf{万博開催前}: 20代9,225人、30代6,341人、40代6,901人、50代12,157人、60代8,247人
\\item \\textbf{万博開催後}: 20代19,666人、30代13,062人、40代16,980人、50代27,927人、60代21,311人
\\item \\textbf{増加率}: 50代129.7\\%増加、60代158.5\\%増加、20代113.1\\%増加
\\end{itemize}

\\textbf{分析示唆}: 万博開催により、特に60代と50代の来訪者増加が顕著で、中高年層の関心が高いことが示唆されます。

\\subsection{時間帯×曜日分析}

時間帯と曜日のクロス集計により、平日と土日祝日の時間帯別来訪パターンを分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{../images/hour_day_heatmap.png}
\\caption{時間帯×曜日来訪者数ヒートマップ}
\\label{fig:hour_day}
\\end{figure}

\\subsection{時間帯×曜日分析結果}
\\begin{itemize}
\\item \\textbf{平日ピーク}: 12時33,099人、13時32,242人、14時30,132人
\\item \\textbf{土日祝日ピーク}: 12時26,168人、13時26,619人、14時26,314人
\\item \\textbf{パターン}: 平日の方が全体的に来訪者数が多く、特に午前中から午後にかけての差が顕著
\\end{itemize}

\\textbf{分析示唆}: 平日の方が土日祝日より来訪者数が多く、通勤・通学などの日常的な来訪が主要な要因であることが確認できます。

\\subsection{エリア×性別分析}

エリアと性別のクロス集計により、エリア別の性別来訪者分布を分析しました。

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{../images/area_gender_heatmap.png}
\\caption{エリア×性別来訪者数ヒートマップ（上位10エリア）}
\\label{fig:area_gender}
\\end{figure}

\\subsection{エリア×性別分析結果}
\\begin{itemize}
\\item \\textbf{万博会場}: 不明216,949人、女性29,747人、男性39,603人
\\item \\textbf{三宮駅}: 不明182,860人、女性24,571人、男性26,726人
\\item \\textbf{姫路駅}: 不明67,516人、女性8,324人、男性9,339人
\\item \\textbf{心斎橋筋商店街}: 不明14,249人、女性1,349人、男性892人
\\item \\textbf{通天閣}: 不明3,280人、女性411人、男性392人
\\end{itemize}

\\textbf{分析示唆}: 万博会場では男性の来訪者が女性を上回り、三宮駅では比較的バランスが取れています。心斎橋筋商店街では女性の来訪者が男性を上回る特徴があります。

\\section{主要な発見}

分析結果から以下の主要な発見が得られました：

\\begin{enumerate}
\\item \\textbf{万博開催の影響}: 万博開催により来訪者数が120.4\\%増加し、特に男性（144.5\\%増加）と60代（158.5\\%増加）で顕著な増加が見られました。

\\item \\textbf{エリアの人気度}: 万博会場（45.5\\%）と三宮駅（37.2\\%）で全体の82.7\\%を占め、エリア間で大きな格差が存在します。

\\item \\textbf{時間帯パターン}: 12-15時にピークを迎え、昼食時間帯を中心とした来訪パターンが明確です。

\\item \\textbf{曜日効果}: 平日（56.2\\%）の方が土日祝日（43.8\\%）よりも来訪者数が多く、通勤・通学などの日常的な来訪が主要な要因です。

\\item \\textbf{人口統計学的特徴}: 性別・年代によって来訪パターンに違いがあり、特に中高年層の関心が高いことが示唆されます。
\\end{enumerate}

\\section{結論}

本分析により、万博開催前後の来訪者データから以下の洞察が得られました：

\\begin{itemize}
\\item \\textbf{万博開催の効果}: 万博開催は来訪者数に大きな影響を与え、特に男性と中高年層の来訪意欲を高めました。

\\item \\textbf{エリア格差}: 万博会場と三宮駅で圧倒的な集客力を示し、エリア間の格差が明確に存在します。

\\item \\textbf{時間帯・曜日パターン}: 平日の昼食時間帯を中心とした来訪パターンが明確で、日常的な来訪が主要な要因です。

\\item \\textbf{人口統計学的特徴}: 性別・年代によって来訪行動に違いがあり、特に男性と中高年層の関心が高いことが示唆されます。
\\end{itemize}

これらの知見は、今後のイベント企画や施設運営に活用できる重要な情報です。

\\section{付録}

\\subsection{生成された画像ファイル}

分析で生成された画像ファイル：
\\begin{itemize}
"""
    
    # 画像ファイル一覧を追加
    for img_file in sorted(image_files):
        img_name = os.path.basename(img_file)
        # アンダースコアをエスケープ
        img_name_escaped = img_name.replace('_', '\\_')
        latex_content += f"\\item {img_name_escaped}\n"
    
    latex_content += """\\end{itemize}

\\end{document}
"""
    
    # LaTeXファイルを保存
    latex_file = os.path.join(reports_dir, "analysis_report_detailed.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"詳細なLaTeXファイルを作成しました: {latex_file}")
    return latex_file

def compile_latex_report(latex_file):
    """LaTeXファイルをコンパイルしてPDFを生成"""
    import subprocess
    import os
    
    reports_dir = os.path.dirname(latex_file)
    os.chdir(reports_dir)
    
    try:
        # xelatexでコンパイル（日本語対応）
        result = subprocess.run(['xelatex', '-interaction=nonstopmode', os.path.basename(latex_file)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("PDFレポートの生成が完了しました")
            pdf_file = os.path.join(reports_dir, "analysis_report_detailed.pdf")
            if os.path.exists(pdf_file):
                print(f"PDFファイル: {pdf_file}")
            else:
                print("PDFファイルの生成に失敗しました")
        else:
            print("LaTeXコンパイルエラー:")
            print(result.stderr)
            
    except FileNotFoundError:
        print("xelatexが見つかりません。LaTeXがインストールされているか確認してください。")
    except Exception as e:
        print(f"PDF生成中にエラーが発生: {e}")

def main():
    """メイン関数"""
    print("詳細なLaTeXレポート生成開始")
    
    # LaTeXファイルを作成
    latex_file = create_detailed_latex_report()
    
    # PDFを生成
    compile_latex_report(latex_file)

if __name__ == "__main__":
    main() 