import streamlit as st
import sympy as sp

# --- MOTEURS MATHÉMATIQUES ---

def gauss_jordan_historique(M):
    A = M.copy()
    lignes, colonnes = A.shape
    etapes = []
    
    etapes.append(("Matrice initiale", A.copy()))
    r = 0 
    
    for c in range(colonnes):
        if r >= lignes:
            break
            
        pivot_row = r
        while pivot_row < lignes and A[pivot_row, c] == 0:
            pivot_row += 1
            
        if pivot_row == lignes:
            continue
            
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
        texte_vp = f"Sous-espace propre pour \\lambda = {sp.latex(vp)} (multiplicité {mult})"
        matrice_vects = sp.Matrix.hstack(*vects)
        etapes.append((texte_vp, matrice_vects))
        
    P, D = M.diagonalize()
    etapes.append(("Matrice de passage P", P.copy()))
    etapes.append(("Matrice diagonale D", D.copy()))
    
    return P, D, etapes

# --- INTERFACE WEB STREAMLIT ---

st.set_page_config(page_title="Solveur Insa", layout="centered")

st.title("🧮 Solveur Automatique de Matrices")

# Le menu de choix
choix = st.selectbox(
    "Que souhaitez-vous faire ?",
    (
        "1. Résoudre un système d'équations (AX = B)",
        "2. Inverser une matrice carrée",
        "3. Réduire une matrice simple",
        "4. Diagonaliser une matrice carrée"
    )
)

st.write("Entrez les coefficients ligne par ligne (séparés par des espaces).")
matrice_input = st.text_area("Votre matrice :", "1 2\n3 4")

if st.button("Lancer le calcul"):
    try:
        # 1. Lecture de la matrice
        lignes_texte = matrice_input.strip().split('\n')
        matrice_liste = [[sp.sympify(val) for val in ligne.split()] for ligne in lignes_texte]
        M = sp.Matrix(matrice_liste)
        lignes, colonnes = M.shape
        
        st.divider()
        st.subheader("Étapes du calcul :")
        
        # 2. Aiguillage selon le choix
        if choix.startswith("1") or choix.startswith("3"):
            M_reduite, historique = gauss_jordan_historique(M)
            
            for description, matrice in historique:
                st.write(f"**{description}**" if "L_" not in str(description) else f"Opération : ${description}$")
                st.latex(sp.latex(matrice))
                
            st.success("Calcul terminé avec succès !")

        elif choix.startswith("2"):
            if lignes != colonnes:
                st.error("ERREUR : La matrice doit être carrée pour être inversée.")
            else:
                M_aug = M.row_join(sp.eye(lignes))
                M_reduite, historique = gauss_jordan_historique(M_aug)
                
                for description, matrice in historique:
                    st.write(f"**{description}**" if "L_" not in str(description) else f"Opération : ${description}$")
                    st.latex(sp.latex(matrice))
                    
                st.success("Calcul terminé ! La matrice inversée se trouve dans la partie droite de la matrice finale.")

        elif choix.startswith("4"):
            if lignes != colonnes:
                st.error("ERREUR : La matrice doit être carrée pour être diagonalisée.")
            else:
                P, D, historique = diagonaliser_historique(M)
                
                for description, data in historique:
                    if isinstance(data, str) and description == "Erreur":
                        st.error(data)
                        break
                        
                    if isinstance(data, sp.Basic):
                        if description == "Polynôme caractéristique":
                            st.write(f"**{description}**")
                            st.latex(sp.latex(data))
                        else:
                            st.write(f"**{description}**")
                            st.latex(sp.latex(data))
                            
                if P is not None and D is not None:
                    st.success("Matrice diagonalisée avec succès !")

    except Exception as e:
        st.error(f"Erreur dans la saisie. Vérifiez vos nombres. (Détail : {e})")

st.divider() 
st.caption("Développé par Evann Carpentier - INSA cvl")