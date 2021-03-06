import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x1 = [76	,
76	,
50	,
76	,
63	,
49	,
78	,
56	,
49	,
61	,
56	,
48	,
76	,
56	,
48	,
61	,
56	,
47	,
60	,
-27	,
-32	,
51	,
89	,
99	,
132	,
97	,
116	,
140	,
91	,
111	,
150	,
92	,
116	,
150	,
92	,
118	,
157	,
96	,
123	,
84	,
94	,
121	,
111	,
94	,
121	,
107	,
94	,
125	,
106	,
95	,
123	,
105	,
94	,
127	,
104	,
95	,
110	,
101	,
96	,
111	,
157	,
171	,
189	,
177	,
176	,
189	,
177	,
177	,
233	,
176	,
178	,
234	,
176	,
179	,
237	,
175	,
181	,
236	,
176	,
182	,
232	,
87	,
52	,
-29	,
44	,
45	,
-32	,
42	,
43	,
-31	,
42	,
46	,
-28	,
44	,
44	,
-32	,
42	,
38	,
-32	,
43	,
40	,
2	,
65	,
95	,
-9	,
64	,
62	,
0	,
68	,
66	,
-2	,
72	,
67	,
3	,
72	,
69	,
3	,
74	,
69	,
8	,
74	,
63	,
3	,
73	,
62	,
8	,
74	,
68	,
65	,
76	,
70	,
46	,
76	,
112	,
55	,
76	,
114	,
61	,
77	,
117	,
50	,
-8	,
-64	,
-59	,
-33	,
-66	,
-57	,
-40	,
-65	,
-54	,
-41	,
-66	,
-55	,
-42	,
-65	,
-55	,
-42	,
-66	,
-54	,
-25	,
-147	,
-136	,
-128	,
-151	,
-136	,
-127	,
-148	,
-132	,
-126	,
-148	,
-132	,
-127	,
-147	,
-131	,
-127	,
-148	,
-130	,
-127	,
-147	,
-130	,
-128	,
-145	,
-128	,
-129	,
-146	,
-127	,
-129	,
-144	,
-126	,
-129	,
-145	,
-125	,
-129	,
-145	,
-124	,
-130	,
-143	,
-124	,
-131	,
-145	,
-123	,
-109	,
58	,
113	,
124	,
132	,
123	,
118	,
132	,
135	,
130	,
145	,
135	,
132	,
143	,
134	,
129	,
163	,
169	,
170	,
190	,
177	,
176	,
196	,
176	,
177	,
196	,
176	,
177	,
189	,
175	,
177	,
189	,
176	,
178	,
191	,
174	,
179	,
190	,
176	,
178	,
188	,
175	,
178	,
193	,
175	,
180	,
191	,
175	,
181	,
189	,
174	,
180	,
192	,
175	,
180	,
190	,
183	,
229	,
241	,
145	,
150	,
152	,
116	,
152	,
147	,
105	,
148	,
98	,
105	,
113	,
102	,
105	,
146	,
103	,
106	,
147	,
105	,
107	,
148	,
105	,
107	,
147	,
103	,
108	,
142	,
108	,
107	,
145	,
106	,
107	,
146	,
108	,
108	,
42	,
106	,
108	,
146	,
107	,
107	,
156	,
108	,
109	,
154	,
109	,
109	,
156	,
-13	,
-14	,
-15	,
-12	,
-11	,
33	,
-14	,
-14	,
35	,
-10	,
-12	,
34	,
-51	,
-119	,
-139	,
-128	,
-124	,
-140	,
-135	,
-132	,
-146	,
-135	,
-133	,
-148	,
-135	,
-135	,
-147	,
-135	,
-134	,
-152	,
-135	,
-137	,
-152	,
-135	,
-137	,
-149	,
-134	,
-138	,
-154	,
-134	,
-139	,
-151	,
-133	,
-140	,
-153	,
-133	,
-142	,
-155	,
-132	,
-141	,
-153	,
-133	,
-150	,
-155	,
-132	,
-151	,
-145	,
-113	,
-134	,
-102	,
-77	,
-97	,
-133	,
-109	,
-132	,
-134	,
-110	,
-131	,
-132	,
-109	,
-132	,
-133	,
-108	,
-102	,
103	,
140	,
274	,
210	,
224	,
236	,
247	,
235	,
242	,
251	,
235	,
245	,
251	,
236	,
243	,
251	,
236	,
181	,
190	,
172	,
180	,
188	,
171	,
179	,
187	,
172	,
181	,
187	,
171	,
178	,
187	,
171	,
180	,
188	,
171	,
180	,
105	,
92	,
102	,
110	,
97	,
109	,
113	,
98	,
107	,
113	,
95	,
106	,
114	,
96	,
107	,
114	,
96	,
108	,
114	,
96	,
107	,
99	,
6	,
-3	,
14	,
-26	,
-9	,
11	,
-28	,
-6	,
7	,
-33	,
-7	,
8	,
-32	,
-7	,
8	,
-33	,
-5	,
8	,
-32	,
54	,
83	,
46	,
76	,
86	,
47	,
84	,
85	,
51	,
83	,
84	,
51	,
82	,
82	,
51	,
82	,
84	,
52	,
83	,
83	,
57	,
80	,
82	,
56	,
78	,
99	,
57	,
83	,
98	,
55	,
81	,
98	,
54	,
81	,
91	,
54	,
80	,
91	,
55	,
78	,
90	,
38	,
-74	,
-4	,
-45	,
-58	,
-34	,
-45	,
-55	,
-32	,
-45	,
-53	,
-31	,
-45	,
-62	,
-32	,
-44	,
-62	,
-32	,
-46	,
-63	,
-37	,
-46	,
-55	,
-31	,
-47	,
-62	,
-35	,
-46	,
-63	,
-34	,
-47	

]
x2=[-15	,
-36	,
-27	,
-14	,
-38	,
-26	,
-16	,
-41	,
-26	,
-23	,
-44	,
-26	,
-20	,
-45	,
-24	,
-26	,
-47	,
-23	,
-29	,
-45	,
-24	,
-34	,
-67	,
-42	,
-60	,
-88	,
-49	,
-73	,
-96	,
-47	,
-79	,
-96	,
-47	,
-78	,
-95	,
-48	,
-79	,
-92	,
-47	,
-159	,
-90	,
-45	,
-125	,
-89	,
-48	,
-114	,
-88	,
-46	,
-128	,
-85	,
-48	,
-118	,
-84	,
-48	,
-125	,
-81	,
-20	,
-125	,
-78	,
-19	,
-116	,
-62	,
-1	,
-108	,
-58	,
2	,
-107	,
-56	,
-66	,
-106	,
-53	,
-65	,
-105	,
-52	,
-69	,
-106	,
-52	,
-67	,
-104	,
-49	,
-65	,
-117	,
-60	,
-101	,
-108	,
-65	,
-101	,
-112	,
-66	,
-104	,
-109	,
-57	,
-98	,
-106	,
-58	,
-101	,
-104	,
-59	,
-100	,
-104	,
-62	,
-94	,
-110	,
-64	,
-104	,
-107	,
-64	,
-102	,
-108	,
-68	,
-107	,
-106	,
-68	,
-103	,
-105	,
-67	,
-102	,
-102	,
-71	,
-104	,
-101	,
-61	,
-108	,
-98	,
-64	,
-116	,
-97	,
-67	,
-121	,
-96	,
-69	,
-136	,
-95	,
-35	,
-132	,
-93	,
-32	,
-130	,
-91	,
-50	,
-132	,
-89	,
-69	,
-95	,
-85	,
-75	,
-104	,
-85	,
-81	,
-103	,
-84	,
-81	,
-103	,
-83	,
-80	,
-103	,
-81	,
-81	,
-101	,
-73	,
-97	,
-129	,
-103	,
-104	,
-126	,
-100	,
-106	,
-126	,
-101	,
-105	,
-125	,
-100	,
-106	,
-125	,
-98	,
-106	,
-124	,
-98	,
-105	,
-126	,
-97	,
-106	,
-123	,
-95	,
-106	,
-125	,
-96	,
-108	,
-124	,
-95	,
-108	,
-122	,
-95	,
-108	,
-122	,
-94	,
-109	,
-122	,
-94	,
-109	,
-121	,
-90	,
-77	,
-81	,
-48	,
-42	,
-60	,
-45	,
-43	,
-66	,
-44	,
-43	,
-66	,
-43	,
-46	,
-65	,
-42	,
-33	,
-52	,
-28	,
-25	,
-53	,
-29	,
-27	,
-50	,
-22	,
-29	,
-49	,
-22	,
-30	,
-48	,
-19	,
-31	,
-49	,
-19	,
-27	,
-46	,
-16	,
-29	,
-46	,
-15	,
-28	,
-44	,
-17	,
-25	,
-44	,
-14	,
-25	,
-44	,
-12	,
-26	,
-42	,
-16	,
-23	,
-44	,
-15	,
-24	,
-43	,
-18	,
-25	,
-55	,
-20	,
-51	,
-64	,
-21	,
-74	,
-62	,
-14	,
-76	,
-59	,
-35	,
-74	,
-56	,
-15	,
-71	,
-55	,
-12	,
-72	,
-53	,
-15	,
-70	,
-50	,
-10	,
-73	,
-48	,
18	,
-68	,
-47	,
-14	,
-70	,
-45	,
-13	,
-70	,
-44	,
-66	,
-69	,
-42	,
-21	,
-67	,
-39	,
-49	,
-67	,
-39	,
-49	,
-68	,
-37	,
-50	,
-91	,
-61	,
-79	,
-90	,
-57	,
-70	,
-96	,
-60	,
-74	,
-92	,
-58	,
-72	,
-95	,
-55	,
-60	,
-73	,
-56	,
-62	,
-71	,
-58	,
-62	,
-74	,
-59	,
-62	,
-75	,
-57	,
-62	,
-72	,
-56	,
-62	,
-73	,
-54	,
-61	,
-70	,
-52	,
-61	,
-70	,
-50	,
-59	,
-69	,
-50	,
-59	,
-68	,
-49	,
-58	,
-67	,
-47	,
-57	,
-66	,
-46	,
-57	,
-65	,
-45	,
-55	,
-64	,
-46	,
-50	,
-60	,
-40	,
-51	,
-59	,
-42	,
-50	,
-58	,
-40	,
-50	,
-58	,
-40	,
-49	,
-57	,
-38	,
-48	,
-54	,
-30	,
-14	,
-22	,
116	,
-1	,
-20	,
3	,
-4	,
-16	,
5	,
0	,
-18	,
7	,
1	,
-17	,
8	,
2	,
-14	,
30	,
25	,
6	,
30	,
25	,
7	,
29	,
25	,
8	,
33	,
27	,
9	,
32	,
26	,
10	,
34	,
29	,
11	,
34	,
30	,
10	,
33	,
22	,
7	,
33	,
23	,
7	,
27	,
24	,
6	,
28	,
22	,
7	,
29	,
22	,
8	,
26	,
24	,
8	,
28	,
24	,
4	,
51	,
40	,
11	,
49	,
40	,
11	,
47	,
43	,
14	,
50	,
41	,
13	,
50	,
43	,
16	,
50	,
42	,
16	,
53	,
58	,
35	,
66	,
61	,
36	,
64	,
64	,
40	,
67	,
66	,
42	,
69	,
67	,
43	,
68	,
65	,
44	,
69	,
68	,
50	,
70	,
68	,
50	,
70	,
2	,
51	,
67	,
5	,
51	,
69	,
6	,
53	,
69	,
31	,
55	,
69	,
33	,
57	,
72	,
35	,
55	,
40	,
33	,
47	,
45	,
31	,
46	,
45	,
29	,
46	,
45	,
31	,
47	,
47	,
32	,
48	,
48	,
32	,
51	,
48	,
32	,
52	,
49	,
31	,
52	,
48	,
32	,
51	,
46	,
32	,
51	

]


X = np.column_stack([x1, x2]) # independent variables

f = [384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
384	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	,
768	
]

#fig = plt.figure()
#ax = fig.gca(projection = '3d')

#ax.plot(x1, x2, f)
#ax.set_xlabel('x1')
#ax.set_ylabel('x2')
#ax.set_zlabel('f(x1,x2)')

#plt.savefig('images/graphical-mulvar-1.png')

#arange = np.linspace(0,5);
#brange = np.linspace(0,5);
#crange = np.linspace(0,5);
#drange = np.linspace(0,5);

#A,B,C,D = np.meshgrid(arange, brange, crange, drange)

def model(X, a,b,c,d,e,f,g,h):
    'Nested function for the model'
    x1 = X[:, 0]
    x2 = X[:, 1]

    f = a+b * x1 + c*x1**2 + d*x1**3 + e*x2 + f*x1*x2 + g*x1**2*x2 + h*x1**3*x2
    return f

@np.vectorize
def errfunc(a, b,c,d,e,f,g,h):
    # function for the summed squared error
    fit = model(X, a, b,c,d,e,f,g,h)
    sse = np.sum((fit - f)**2)
    return sse

#SSE = errfunc(A, B, C, D)

#plt.clf()
#plt.contourf(A, B, SSE, 50)
#plt.plot([3.2], [2.1], 'ro')
#plt.figtext( 3.4, 2.2, 'Minimum near here', color='r')

#plt.savefig('images/graphical-mulvar-2.png')

guesses = [3.18, 2.02,65,89,3.18, 2.02,65,89]
#guesses =  [ 1.07719186e+03 , 6.25984134e+00 ,-2.74564401e-02, -1.16868713e-04, 
  #1.06701813e+00, -2.77235878e-02, -3.60254567e-05 , 7.80869135e-07]


from scipy.optimize import curve_fit

popt, pcov = curve_fit(model, X, f, guesses)
print(popt)

#plt.plot([popt[0]], [popt[1]], 'r*')
#plt.savefig('images/graphical-mulvar-3.png')

print(model(X, *popt))

#fig = plt.figure()
#ax = fig.gca(projection = '3d')

#ax.plot(x1, x2, f, 'ko', label='data')
#ax.plot(x1, x2, model(X, *popt), 'r-', label='fit')
#ax.set_xlabel('x1')
#ax.set_ylabel('x2')
#ax.set_zlabel('f(x1,x2)')

#plt.show()