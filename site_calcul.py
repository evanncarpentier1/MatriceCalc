import streamlit as st
import sympy as sp
import sympy.stats as sp_stats

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Solveur Insa", layout="centered")

# --- CSS POUR UN PDF PARFAIT (Ctrl+P) ---
st.markdown("""
    <style>
    @media print {
        header, .stSidebar, .stSelectbox, .stTextArea, .stTextInput, .stButton, footer, .stAlert, .stInfo, .stRadio {
            display: none !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- MOTEURS MATHÉMATIQUES (Algèbre) ---
def gauss_jordan_historique(M):
    A = M.copy()
    lignes, colonnes = A.shape
    etapes = []
    etapes.append(("Matrice initiale", A.copy()))
    r = 0 
    for c in range(colonnes):
        if r >= lignes: break
        pivot_row = r
        while pivot_row < lignes and A[pivot_row, c] == 0:
            pivot_row += 1
        if pivot_row == lignes: continue
        if pivot_row != r:
            A.row_swap(r, pivot_row)
            op = f"L_{r+1} \\leftrightarrow L_{pivot_row+1}"
            etapes.append((op, A.copy()))
        pivot_val = A[r, c]
        if pivot_val != 1:
            A.row_op(r, lambda x, j: x / pivot_val)
            op = f"L_{r+1} \\leftarrow \\frac{{1}}{{{sp.latex(pivot_val)}}} L_{r+1}"
            etapes.append((op, A.copy()))
        for i in range(lignes):
            if i != r and A[i, c] != 0:
                facteur = A[i, c]
                A.row_op(i, lambda x, j: x - facteur * A[r, j])
                signe = "-" if facteur > 0 else "+"
                val_absolue = sp.latex(abs(facteur))
                op = f"L_{i+1} \\leftarrow L_{i+1} {signe} {val_absolue} L_{r+1}"
                etapes.append((op, A.copy()))
        r += 1
    return A, etapes

def diagonaliser_historique(M):
    etapes = []
    etapes.append(("Matrice initiale A", M.copy()))
    lam = sp.Symbol('\\lambda')
    poly = M.charpoly(lam)
    eq_poly = sp.Eq(poly.as_expr(), 0)
    etapes.append(("Polynôme caractéristique", eq_poly))
    if not M.is_diagonalizable():
        etapes.append(("Erreur", "La matrice n'est pas diagonalisable."))
        return None, None, etapes
    vecteurs_propres = M.eigenvects()
    for vp, mult, vects in vecteurs_propres:
        texte_vp = f"Sous-espace propre pour $\\lambda = {sp.latex(vp)}$ (multiplicité {mult})"
        matrice_vects = sp.Matrix.hstack(*vects)
        etapes.append((texte_vp, matrice_vects))
    P, D = M.diagonalize()
    etapes.append(("Matrice de passage P", P.copy()))
    etapes.append(("Matrice diagonale D", D.copy()))
    return P, D, etapes

# ==========================================
# BARRE LATÉRALE : NAVIGATION
# ==========================================
st.sidebar.title("🧮 Navigation")
section = st.sidebar.radio(
    "Choisis le domaine :",
    [
        "1. Algèbre Linéaire", 
        "2. Analyse Mathématique", 
        "3. Équations Différentielles", 
        "4. Géométrie (Coniques)",
        "5. Nombres Complexes",
        "6. Transformées (Laplace & Fourier)",
        "7. Calcul Vectoriel",
        "8. Séries Numériques",
        "9. Probabilités & Statistiques"
    ]
)

st.sidebar.divider()
st.sidebar.info("🖨️ **Astuce PDF :** Fais `Ctrl + P` sur n'importe quelle page de résultats.")

# Variables globales pour SymPy
x, y, z = sp.symbols('x y z')
t, w, p = sp.symbols('t w p', real=True)
n, k = sp.symbols('n k', integer=True)


# ==========================================
# SECTION 1 : ALGÈBRE LINÉAIRE
# ==========================================
if section.startswith("1"):
    st.title("🧮 Algèbre Linéaire")
    choix = st.selectbox("Opération :", ("Résoudre un système (AX = B)", "Inverser une matrice", "Réduire une matrice", "Diagonaliser", "Noyau (Ker) et Image (Im)"))
    matrice_input = st.text_area("Coefficients ligne par ligne (espacés) :", "1 2\n3 4")

    if st.button("Calculer la matrice"):
        try:
            lignes_texte = matrice_input.strip().split('\n')
            matrice_liste = [[sp.sympify(val) for val in ligne.split()] for ligne in lignes_texte]
            M = sp.Matrix(matrice_liste)
            lignes, colonnes = M.shape
            st.divider()
            
            if choix.startswith("Résoudre") or choix.startswith("Réduire"):
                M_reduite, historique = gauss_jordan_historique(M)
                for desc, mat in historique:
                    st.write(f"**{desc}**" if "L_" not in str(desc) else f"Opération : ${desc}$")
                    st.latex(sp.latex(mat))
            elif choix.startswith("Inverser"):
                if lignes != colonnes: st.error("La matrice doit être carrée.")
                else:
                    M_aug = M.row_join(sp.eye(lignes))
                    M_reduite, historique = gauss_jordan_historique(M_aug)
                    for desc, mat in historique:
                        st.write(f"**{desc}**" if "L_" not in str(desc) else f"Opération : ${desc}$")
                        st.latex(sp.latex(mat))
            elif choix.startswith("Diagonaliser"):
                if lignes != colonnes: st.error("La matrice doit être carrée.")
                else:
                    P, D, historique = diagonaliser_historique(M)
                    for desc, data in historique:
                        if desc == "Erreur": st.error(data); break
                        st.write(f"**{desc}**")
                        st.latex(sp.latex(data))
            elif choix.startswith("Noyau"):
                st.latex(sp.latex(M))
                ker_vects = M.nullspace()
                st.write("**Base du Noyau** $\\ker(A)$ **:**")
                if not ker_vects: st.write("Réduit au vecteur nul $\\{0\\}$. L'application est injective.")
                else: st.latex(sp.latex(sp.Matrix.hstack(*ker_vects)))
                im_vects = M.columnspace()
                st.write("**Base de l'Image** $\\text{Im}(A)$ **:**")
                if not im_vects: st.write("Réduite au vecteur nul.")
                else: st.latex(sp.latex(sp.Matrix.hstack(*im_vects)))
        except Exception as e: st.error(f"Erreur de syntaxe : {e}")

# ==========================================
# SECTION 2 : ANALYSE MATHÉMATIQUE
# ==========================================
elif section.startswith("2"):
    st.title("📈 Analyse Mathématique")
    choix_analyse = st.selectbox("Opération :", ["Dérivée", "Primitive", "Intégrale définie", "Développement Limité", "Limite"])
    func_str = st.text_input("Fonction $f(x)$ :", "x**2 * exp(-x)")

    borne_inf, borne_sup, pt_limite, ordre_dl = None, None, "0", 3
    if "définie" in choix_analyse:
        c1, c2 = st.columns(2)
        borne_inf = c1.text_input("De :", "0")
        borne_sup = c2.text_input("À :", "oo")
    elif "Développement" in choix_analyse:
        c1, c2 = st.columns(2)
        pt_limite = c1.text_input("Autour de :", "0")
        ordre_dl = c2.number_input("Ordre :", value=3)
    elif "Limite" in choix_analyse:
        pt_limite = st.text_input("Quand x tend vers :", "0")

    if st.button("Analyser"):
        try:
            f = sp.sympify(func_str)
            st.divider()
            st.latex(f"f(x) = {sp.latex(f)}")
            if "Dérivée" in choix_analyse:
                res = sp.diff(f, x)
                st.latex(f"f'(x) = {sp.latex(res)} = {sp.latex(sp.simplify(res))}")
            elif "Primitive" in choix_analyse:
                st.latex(f"F(x) = {sp.latex(sp.integrate(f, x))} + C")
            elif "définie" in choix_analyse:
                b_inf, b_sup = sp.sympify(borne_inf), sp.sympify(borne_sup)
                res = sp.integrate(f, (x, b_inf, b_sup))
                st.latex(f"\\int_{{{sp.latex(b_inf)}}}^{{{sp.latex(b_sup)}}} ({sp.latex(f)}) dx = {sp.latex(res)}")
            elif "Développement" in choix_analyse:
                pt = sp.sympify(pt_limite)
                res = sp.series(f, x, pt, ordre_dl).removeO()
                st.latex(f"DL_{{{ordre_dl}}}({sp.latex(pt)}) = {sp.latex(res)} + o((x - {sp.latex(pt)})^{{{ordre_dl}}})")
            elif "Limite" in choix_analyse:
                pt = sp.sympify(pt_limite)
                st.latex(f"\\lim_{{x \\to {sp.latex(pt)}}} f(x) = {sp.latex(sp.limit(f, x, pt))}")
        except Exception as e: st.error(f"Erreur : {e}")

# ==========================================
# SECTION 3 : ÉQUATIONS DIFFÉRENTIELLES
# ==========================================
elif section.startswith("3"):
    st.title("⚙️ Équations Différentielles")
    st.info("💡 **Syntaxe intuitive :** Utilise `y` pour $y(x)$, `y'` pour la dérivée première, et `y''` pour la dérivée seconde.")
    
    col1, col2 = st.columns([2, 1])
    eq_gauche = col1.text_input("Membre de gauche :", "y'' + 2*y' + y")
    eq_droite = col2.text_input("Membre de droite :", "exp(-x)")
    
    if st.button("Résoudre l'EDO"):
        try:
            g_temp = eq_gauche.replace("y''", "D2").replace("y'", "D1").replace("y", "D0")
            d_temp = eq_droite.replace("y''", "D2").replace("y'", "D1").replace("y", "D0")
            
            g_str = g_temp.replace("D2", "Derivative(YFUNC(x), (x, 2))").replace("D1", "Derivative(YFUNC(x), x)").replace("D0", "YFUNC(x)")
            d_str = d_temp.replace("D2", "Derivative(YFUNC(x), (x, 2))").replace("D1", "Derivative(YFUNC(x), x)").replace("D0", "YFUNC(x)")
            
            locs = {"YFUNC": sp.Function('y')}
            expr_gauche = sp.sympify(g_str, locals=locs)
            expr_droite = sp.sympify(d_str, locals=locs)
            
            eq = sp.Eq(expr_gauche, expr_droite)
            st.divider()
            st.write("**Équation interprétée :**")
            st.latex(sp.latex(eq))
            st.write("**Solution générale :**")
            st.latex(sp.latex(sp.dsolve(eq)))
        except Exception as e: st.error(f"Erreur de syntaxe : {e}")

# ==========================================
# SECTION 4 : GÉOMÉTRIE (Coniques avec détails)
# ==========================================
elif section.startswith("4"):
    st.title("📐 Géométrie (Coniques)")
    eq_input = st.text_input("Équation $P(x,y) = 0$ :", "5*x**2 - 4*x*y + 8*y**2 - 36")
    if st.button("Analyser la conique"):
        try:
            expr = sp.sympify(eq_input)
            st.divider()
            
            st.write("**1. Équation de départ :**")
            st.latex(f"P(x, y) = {sp.latex(expr)} = 0")

            st.write("**2. Calcul des dérivées partielles secondes :**")
            dx2 = sp.diff(expr, x, 2)
            dy2 = sp.diff(expr, y, 2)
            dxy = sp.diff(expr, x, y)
            
            st.latex(f"\\frac{{\\partial^2 P}}{{\\partial x^2}} = {sp.latex(dx2)} \\implies a = {sp.latex(dx2/2)}")
            st.latex(f"\\frac{{\\partial^2 P}}{{\\partial y^2}} = {sp.latex(dy2)} \\implies c = {sp.latex(dy2/2)}")
            st.latex(f"\\frac{{\\partial^2 P}}{{\\partial x \\partial y}} = {sp.latex(dxy)} \\implies b = {sp.latex(dxy/2)}")

            a, b, c = dx2/2, dxy/2, dy2/2
            A = sp.Matrix([[a, b], [b, c]])
            det_A = A.det()
            
            st.write("**3. Matrice de la forme quadratique et Déterminant :**")
            st.latex(f"A = {sp.latex(A)}")
            st.latex(f"\\det(A) = ({sp.latex(a)}) \\times ({sp.latex(c)}) - ({sp.latex(b)})^2 = {sp.latex(det_A)}")
            
            nature = "Ellipse" if det_A > 0 else "Hyperbole" if det_A < 0 else "Parabole"
            st.write(f"**Conclusion :** Le déterminant est {'strictement positif' if det_A > 0 else 'strictement négatif' if det_A < 0 else 'nul'}, il s'agit donc d'une forme de type **{nature}**.")

            if det_A != 0:
                st.write("**4. Recherche du centre $\\Omega$ (Annulation du gradient) :**")
                dx = sp.diff(expr, x)
                dy = sp.diff(expr, y)
                st.latex(f"\\frac{{\\partial P}}{{\\partial x}} = {sp.latex(dx)} = 0")
                st.latex(f"\\frac{{\\partial P}}{{\\partial y}} = {sp.latex(dy)} = 0")
                centre = sp.solve((sp.Eq(dx, 0), sp.Eq(dy, 0)), (x, y))
                if centre: 
                    st.latex(f"\\Omega \\left( {sp.latex(centre[x])}, {sp.latex(centre[y])} \\right)")

            st.write("**5. Valeurs propres $\\lambda$ (Signature géométrique) :**")
            lam = sp.Symbol('\\lambda')
            poly = A.charpoly(lam)
            st.latex(f"\\det(A - \\lambda I) = {sp.latex(poly.as_expr())} = 0")
            for vp in A.eigenvals(): 
                st.latex(f"\\lambda = {sp.latex(vp)}")
                
        except Exception as e: st.error(f"Erreur : {e}")

# ==========================================
# SECTION 5 : NOMBRES COMPLEXES
# ==========================================
elif section.startswith("5"):
    st.title("⚡ Nombres Complexes")
    choix_complexe = st.selectbox("Calcul :", ["Analyse (Module, Arg)", "Racines n-ièmes", "Équation complexe", "Impédance équivalente"])
    
    if choix_complexe.startswith("Analyse") or choix_complexe.startswith("Racines"):
        z_input = st.text_input("Complexe $z$ (i ou j) :", "1 + i")
        n_racines = st.number_input("Ordre n :", value=3) if choix_complexe.startswith("Racines") else None
        
        if st.button("Calculer"):
            try:
                Z = sp.sympify(z_input.replace('i', 'I').replace('j', 'I'))
                st.divider()
                st.write("**Nombre $z$ saisi :**")
                st.latex(f"z = {sp.latex(Z)}")
                
                mod, arg = sp.simplify(sp.Abs(Z)), sp.simplify(sp.arg(Z))
                
                if choix_complexe.startswith("Analyse"):
                    st.write("**1. Calcul du Module :**")
                    st.latex(f"|z| = \\sqrt{{\\text{{Re}}(z)^2 + \\text{{Im}}(z)^2}} = {sp.latex(mod)}")
                    
                    st.write("**2. Calcul de l'Argument principal :**")
                    st.latex(f"\\arg(z) = {sp.latex(arg)} \\pmod{{2\\pi}}")
                    
                    st.write("**3. Forme exponentielle finale :**")
                    st.latex(f"z = {sp.latex(mod)} e^{{i \\left({sp.latex(arg)}\\right)}}")
                else:
                    st.write(f"**Formule des racines {n_racines}-ièmes :**")
                    st.latex(f"z_k = \\sqrt[{n_racines}]{{{sp.latex(mod)}}} e^{{i \\frac{{{sp.latex(arg)} + 2k\\pi}}{{{n_racines}}}}}")
                    st.write("**Résultats simplifiés :**")
                    for idx, r in enumerate(sp.solve(x**n_racines - Z, x)):
                        st.latex(f"z_{idx} = {sp.latex(sp.simplify(r))}")
            except Exception as e: st.error(f"Erreur : {e}")
            
    elif choix_complexe.startswith("Équation"):
        eq_input = st.text_input("Équation (ex: z**2 + z + 1 = 0) :", "z**2 + (1+i)*z - 2")
        if st.button("Résoudre"):
            try:
                eq_c = eq_input.replace('i', 'I').replace('j', 'I')
                expr = sp.sympify(eq_c.split("=")[0]) - sp.sympify(eq_c.split("=")[1]) if "=" in eq_c else sp.sympify(eq_c)
                st.divider()
                st.write("**Équation traitée :**")
                st.latex(f"{sp.latex(expr)} = 0")
                st.write("**Solutions :**")
                for sol in sp.solve(expr, z): st.latex(f"z = {sp.latex(sp.simplify(sol))}")
            except Exception as e: st.error(f"Erreur : {e}")
            
    elif choix_complexe.startswith("Impédance"):
        type_asso = st.radio("Association :", ["Série", "Parallèle"])
        z_list = st.text_area("Impédances (séparées par des virgules) :", "100, 50*i")
        if st.button("Calculer Z_eq"):
            try:
                imps = [sp.sympify(i.strip().replace('i', 'I').replace('j', 'I')) for i in z_list.split(',')]
                st.divider()
                if type_asso == "Série":
                    st.write("**Somme des impédances (Série) :**")
                    st.latex(f"Z_{{eq}} = " + " + ".join([sp.latex(i) for i in imps]))
                    Z_eq = sum(imps)
                else:
                    st.write("**Somme des admittances (Parallèle) :**")
                    st.latex(f"\\frac{{1}}{{Z_{{eq}}}} = " + " + ".join([f"\\frac{{1}}{{{sp.latex(i)}}}" for i in imps]))
                    Z_eq = 1 / sum([1/i for i in imps])
                
                st.write("**Impédance équivalente finale :**")
                st.latex(f"Z_{{eq}} = {sp.latex(sp.simplify(Z_eq))}")
            except Exception as e: st.error(f"Erreur : {e}")

# ==========================================
# SECTION 6 : TRANSFORMÉES (Laplace & Fourier)
# ==========================================
elif section.startswith("6"):
    st.title("🔀 Transformées")
    choix_transfo = st.selectbox("Opération :", ["Laplace", "Laplace Inverse", "Fourier", "Fourier Inverse"])
    
    if "Inverse" not in choix_transfo:
        func_str = st.text_input("Fonction temporelle $f(t)$ :", "exp(-2*t) * Heaviside(t)")
    else:
        var = "p" if "Laplace" in choix_transfo else "w"
        func_str = st.text_input(f"Fonction fréquentielle $F({var})$ :", f"1 / ({var} + 2)")

    if st.button("Calculer la Transformée"):
        try:
            locs = {"t": t, "p": p, "w": w}
            expr = sp.sympify(func_str, locals=locs)
            st.divider()
            if "Laplace" in choix_transfo and "Inverse" not in choix_transfo:
                st.latex(f"\\mathcal{{L}}\\{{{sp.latex(expr)}\\}}(p) = {sp.latex(sp.laplace_transform(expr, t, p, noconds=True))}")
            elif "Laplace Inverse" in choix_transfo:
                st.latex(f"\\mathcal{{L}}^{{-1}}\\{{{sp.latex(expr)}\\}}(t) = {sp.latex(sp.inverse_laplace_transform(expr, p, t))}")
            elif "Fourier" in choix_transfo and "Inverse" not in choix_transfo:
                st.latex(f"\\mathcal{{F}}\\{{{sp.latex(expr)}\\}}(w) = {sp.latex(sp.fourier_transform(expr, t, w))}")
            elif "Fourier Inverse" in choix_transfo:
                st.latex(f"\\mathcal{{F}}^{{-1}}\\{{{sp.latex(expr)}\\}}(t) = {sp.latex(sp.inverse_fourier_transform(expr, w, t))}")
        except Exception as e: st.error(f"Erreur : {e}")

# ==========================================
# SECTION 7 : CALCUL VECTORIEL (Avec détails)
# ==========================================
elif section.startswith("7"):
    st.title("🧭 Calcul Vectoriel")
    choix_vect = st.selectbox("Opérateur :", ["Gradient", "Divergence", "Rotationnel"])

    if "Gradient" in choix_vect:
        f_str = st.text_input("Champ $f(x, y, z)$ :", "x**2 * y + z**3")
        if st.button("Calculer le Gradient"):
            try:
                f = sp.sympify(f_str)
                st.divider()
                st.write("**1. Calcul des dérivées partielles :**")
                dx, dy, dz = sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)
                st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(dx)}")
                st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(dy)}")
                st.latex(f"\\frac{{\\partial f}}{{\\partial z}} = {sp.latex(dz)}")
                
                st.write("**2. Vecteur Gradient final :**")
                st.latex(sp.latex(sp.Matrix([dx, dy, dz])))
            except Exception as e: st.error(f"Erreur : {e}")
    else:
        c1, c2, c3 = st.columns(3)
        vx_str = c1.text_input("$V_x$ :", "x*y")
        vy_str = c2.text_input("$V_y$ :", "y*z")
        vz_str = c3.text_input("$V_z$ :", "x*z")
        if st.button("Calculer"):
            try:
                Vx, Vy, Vz = sp.sympify(vx_str), sp.sympify(vy_str), sp.sympify(vz_str)
                st.divider()
                
                if "Divergence" in choix_vect:
                    st.write("**1. Calcul des dérivées partielles directes :**")
                    dx, dy, dz = sp.diff(Vx, x), sp.diff(Vy, y), sp.diff(Vz, z)
                    st.latex(f"\\frac{{\\partial V_x}}{{\\partial x}} = {sp.latex(dx)}, \\quad \\frac{{\\partial V_y}}{{\\partial y}} = {sp.latex(dy)}, \\quad \\frac{{\\partial V_z}}{{\\partial z}} = {sp.latex(dz)}")
                    st.write("**2. Divergence (Somme) :**")
                    st.latex(f"\\vec{{\\nabla}} \\cdot \\vec{{V}} = {sp.latex(dx + dy + dz)}")
                else:
                    st.write("**1. Calcul des composantes croisées :**")
                    rot_x = sp.diff(Vz, y) - sp.diff(Vy, z)
                    rot_y = sp.diff(Vx, z) - sp.diff(Vz, x)
                    rot_z = sp.diff(Vy, x) - sp.diff(Vx, y)
                    st.latex(f"\\vec{{u}}_x : \\frac{{\\partial V_z}}{{\\partial y}} - \\frac{{\\partial V_y}}{{\\partial z}} = {sp.latex(rot_x)}")
                    st.latex(f"\\vec{{u}}_y : \\frac{{\\partial V_x}}{{\\partial z}} - \\frac{{\\partial V_z}}{{\\partial x}} = {sp.latex(rot_y)}")
                    st.latex(f"\\vec{{u}}_z : \\frac{{\\partial V_y}}{{\\partial x}} - \\frac{{\\partial V_x}}{{\\partial y}} = {sp.latex(rot_z)}")
                    st.write("**2. Vecteur Rotationnel final :**")
                    st.latex(sp.latex(sp.Matrix([rot_x, rot_y, rot_z])))
            except Exception as e: st.error(f"Erreur : {e}")

# ==========================================
# SECTION 8 : SÉRIES NUMÉRIQUES (Avec détails)
# ==========================================
elif section.startswith("8"):
    st.title("♾️ Séries Numériques")
    col1, col2 = st.columns([3, 1])
    un_str = col1.text_input("Terme général $u_n$ :", "2 * (3/4)^n")
    n0_str = col2.number_input("Indice de départ $n_0$ :", value=0)
    
    if st.button("Calculer la Somme"):
        try:
            un_clean = un_str.replace('^', '**')
            locs = {"n": n}
            un = sp.sympify(un_clean, locals=locs)
            st.divider()
            
            st.write("**1. Évaluation des premiers termes :**")
            n0 = int(n0_str)
            termes = [sp.simplify(un.subs(n, n0 + i)) for i in range(3)]
            st.latex(f"u_{{{n0}}} = {sp.latex(termes[0])}, \\quad u_{{{n0+1}}} = {sp.latex(termes[1])}, \\quad u_{{{n0+2}}} = {sp.latex(termes[2])}")
            st.latex(f"\\sum = {sp.latex(termes[0])} + {sp.latex(termes[1])} + {sp.latex(termes[2])} + \\dots")
            
            st.write("**2. Résultat de la série formelle :**")
            somme_formelle = sp.Sum(un, (n, n0, sp.oo))
            resultat = somme_formelle.doit()
            st.latex(f"\\sum_{{n={n0}}}^\\infty {sp.latex(un)} = {sp.latex(resultat)}")
            
            if "Sum" in str(resultat): st.warning("⚠️ SymPy n'a pas pu évaluer cette somme (divergence possible ou pas de forme close connue).")
        except Exception as e: st.error(f"Erreur : {e}")

# ==========================================
# SECTION 9 : PROBABILITÉS & STATS
# ==========================================
elif section.startswith("9"):
    st.title("🎲 Probabilités & Statistiques")
    loi = st.selectbox("Loi de probabilité :", ["1. Normale", "2. Exponentielle", "3. Binomiale", "4. Poisson"])
    
    params = {}
    if loi.startswith("1"):
        c1, c2 = st.columns(2)
        params['mu'], params['sigma'] = c1.text_input("Espérance $\\mu$ :", "0"), c2.text_input("Écart-type $\\sigma$ :", "1")
    elif loi.startswith("2"):
        params['lambda'] = st.text_input("Taux $\\lambda$ (> 0) :", "2")
    elif loi.startswith("3"):
        c1, c2 = st.columns(2)
        params['n'], params['p'] = c1.text_input("Nombre d'essais $n$ :", "10"), c2.text_input("Probabilité $p$ :", "1/2")
    elif loi.startswith("4"):
        params['lambda'] = st.text_input("Paramètre $\\lambda$ :", "3")
        
    if st.button("Analyser la Loi"):
        try:
            st.divider()
            if loi.startswith("1"):
                X = sp_stats.Normal('X', sp.sympify(params['mu']), sp.sympify(params['sigma']))
                st.write("**Loi Normale :** Densité $f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}$")
                st.latex(f"f(x) = {sp.latex(sp_stats.density(X)(x))}")
            elif loi.startswith("2"):
                X = sp_stats.Exponential('X', sp.sympify(params['lambda']))
                st.write("**Loi Exponentielle :** Densité $f(x) = \\lambda e^{-\\lambda x}$")
                st.latex(f"f(x) = {sp.latex(sp_stats.density(X)(x))}")
            elif loi.startswith("3"):
                X = sp_stats.Binomial('X', sp.sympify(params['n']), sp.sympify(params['p']))
                st.write("**Loi Binomiale :** $P(X=k) = \\binom{n}{k} p^k (1-p)^{n-k}$")
                st.latex(f"P(X = k) = {sp.latex(sp_stats.density(X)(k))}")
            elif loi.startswith("4"):
                X = sp_stats.Poisson('X', sp.sympify(params['lambda']))
                st.write("**Loi de Poisson :** $P(X=k) = \\frac{\\lambda^k}{k!} e^{-\\lambda}$")
                st.latex(f"P(X = k) = {sp.latex(sp_stats.density(X)(k))}")
            
            st.write("**Propriétés de la Variable Aléatoire $X$ :**")
            st.latex(f"E(X) = {sp.latex(sp.simplify(sp_stats.E(X)))}")
            st.latex(f"V(X) = {sp.latex(sp.simplify(sp_stats.variance(X)))}")
            
        except Exception as e: st.error(f"Erreur d'analyse. Assure-toi que les paramètres sont cohérents. (Détail : {e})")

st.sidebar.divider()
st.sidebar.caption("Développé par Evann Carpentier - INSA cvl")
